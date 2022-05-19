import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from condgen.utils import str2bool
import math
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PositionalEncoding():
    """Apply positional encoding to instances."""

    def __init__(self, min_timescale, max_timescale, n_channels):
        """PositionalEncoding.
        Args:
            min_timescale: minimal scale of values
            max_timescale: maximal scale of values
            n_channels: number of channels to use to encode position
        """
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.n_channels = n_channels

        self._num_timescales = self.n_channels // 2
        self._inv_timescales = self._compute_inv_timescales()

    def _compute_inv_timescales(self):
        log_timescale_increment = (
            math.log(float(self.max_timescale) / float(self.min_timescale))
            / (float(self._num_timescales) - 1)
        )
        inv_timescales = (
            self.min_timescale
            * np.exp(
                np.arange(self._num_timescales)
                * -log_timescale_increment
            )
        )
        return torch.Tensor(inv_timescales)

    def __call__(self, times):
        """Apply positional encoding to instances."""
        # instance = instance.copy()  # We only want a shallow copy
        positions = times
        scaled_time = (
            positions[...,None] *
            self._inv_timescales[None, :].to(times.device)
        )
        signal = torch.cat(
            (torch.sin(scaled_time), torch.cos(scaled_time)),
            axis=-1
        )
        return signal

class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You
    Need".  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention
    is all you need. In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement in a different way during
    application.
    This class is adapted from the pytorch source code.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
            (default=2048).
        dropout: the dropout value (default=0.1).
        norm: Normalization to apply, one of 'layer' or 'rezero'.
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 norm='layer',  nhead_treat = 0, treat_attention_type = None, linear_type = "normal"):
        super(TransformerEncoderLayer, self).__init__()
        if norm == 'layer':
            def get_residual():
                def residual(x1, x2):
                    return x1 + x2
                return residual

            def get_norm():
                return nn.LayerNorm(d_model)
        elif norm == 'rezero':
            def get_residual():
                return ReZero()

            def get_norm():
                return nn.Identity()
        else:
            raise ValueError('Invalid normalization: {}'.format(norm))
        
        self.nhead = nhead
        self.nhead_treat = nhead_treat
        
        d_model_head = d_model // (nhead + nhead_treat)
        self.d_model_head = d_model_head

        self.map_embed = nn.Linear(d_model, d_model_head*nhead)
        self.self_attn = nn.MultiheadAttention( d_model_head * nhead, nhead, dropout=dropout)
        
        if self.nhead_treat>0:
            self.map_embed_treat = nn.Linear(d_model, d_model_head*nhead_treat)
            self.map_embed_treat_src = nn.Linear(d_model, d_model_head*nhead_treat)
            self.treat_attn = nn.MultiheadAttention(d_model_head * nhead_treat, nhead_treat, dropout = dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = get_norm()
        self.norm2 = get_norm()
        self.residual1 = get_residual()
        self.residual2 = get_residual()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self.treat_attention_type = treat_attention_type

        if self.treat_attention_type == "expert":
            self.alpha1 = nn.Parameter(torch.rand(1))
            self.alpha2 = nn.Parameter(torch.rand(1))
            self.alpha3 = nn.Parameter(torch.rand(1))
            self.gamma = nn.Parameter(torch.rand(1))
            self.b0 = nn.Parameter(torch.rand(1))
            self.bl = nn.Parameter(torch.rand(1))

        elif self.treat_attention_type =="linear":
            if linear_type == "normal":
                self.querry_mod = nn.Linear(d_model,d_model)
            elif linear_type == "all":
                self.querry_mod = nn.Linear(2*d_model,d_model)

            self.key_mod = nn.Linear(2*d_model,d_model)
            self.value_mod = nn.Linear(2*d_model,d_model)

            self.linear_type = linear_type

        elif self.treat_attention_type == "mlp":
            self.mlp1 = nn.Sequential(nn.Linear(1,50),nn.ReLU(),nn.Linear(50,d_model))
            self.mlp2 = nn.Sequential(nn.Linear(1,50),nn.ReLU(),nn.Linear(50,d_model))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)


    def g3(self,src,src_treat):
        T = src.shape[0]
        N = src.shape[1]
        time_diffs = torch.nn.functional.relu(torch.arange(T,device=src.device)[None,:]-torch.arange(T,device = src.device)[:,None])[None,...].repeat(N,1,1)

        alpha1 = self.alpha1
        alpha2 = self.alpha2
        alpha3 = self.alpha3
        gamma = self.gamma
        alpha0 = (alpha1 + 2*self.b0 - self.bl) / (1+torch.exp(-alpha3*gamma/2))
        before = self.b0 + alpha1/(1+torch.exp(-alpha2*(time_diffs-gamma/2)))
        after  = self.bl + alpha0/(1+torch.exp(alpha3*(time_diffs-2*gamma/2)))
        
        before[time_diffs>=gamma] = after[time_diffs>=gamma]
        before[time_diffs==0] = 0

        activations = torch.matmul(src_treat[:,-1,:][:,None,:],before) # Selecting the -1 dimension whih corresponds to the therapy change.

        return activations


    def mlp_head(self,src_treat, src_mask_treat):
        T = src_treat.shape[0]
        N = src_treat.shape[1]
        time_diffs = (torch.arange(T,device = src_treat.device)[:,None] - torch.arange(T, device = src_treat.device)[None,:])[None,...].repeat(N,1,1).float()

        mask = (~src_mask_treat).float()[:N]

        treat_vec1 = src_treat[...,0].permute(1,0)[...,None] 
        treat_vec2 = src_treat[...,1].permute(1,0)[...,None] 
        
        mask1 = mask * treat_vec1
        mask2 = mask * treat_vec2

        effect1 = self.mlp1(time_diffs[...,None])
        effect2 = self.mlp2(time_diffs[...,None])
        
        additive1 = (effect1 * mask1[...,None]).sum(-2)
        additive2 = (effect2 * mask2[...,None]).sum(-2)
        
        return (additive1 + additive2).permute(1,0,2)

    def forward(self, src, src_treat, src_mask=None, src_mask_treat = None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required). # T X N X D
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        
        if len(src_mask.shape)==3:
            src_mask = src_mask.repeat(self.nhead,1,1)
            if self.treat_attention_type == "mlp":
                src_mask_treat = src_mask_treat
            else:
                src_mask_treat = src_mask_treat.repeat(self.nhead_treat,1,1)
        
        if self.treat_attention_type=="linear":
            if self.linear_type=="normal":
                querry = self.querry_mod(src)
            elif self.linear_type == "all":
                querry = self.querry_mod(torch.cat((src,src_treat),-1))
            else:
                raise("linear type unknown")

            key = self.key_mod(torch.cat((src,src_treat),-1))
            values = self.value_mod(torch.cat((src,src_treat),-1))
            
            src2 = self.self_attn(
            querry, key, values,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        else:
            mapped_src = self.map_embed(src)
            src2 = self.self_attn(
            mapped_src, mapped_src, mapped_src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        #src2 = torch.nan_to_num(src2)
        if self.nhead_treat>0:
            if self.treat_attention_type == "expert":
                activations = self.g3(src,src_treat).permute(2,0,1)
                src2 = src2 + activations

            else:
                mapped_src_ = self.map_embed_treat_src(src)
                mapped_src_treat = self.map_embed_treat(src_treat)
                
                src_treat2 =  self.treat_attn(
                mapped_src_, mapped_src_treat, mapped_src_treat,
                attn_mask=src_mask_treat,
                key_padding_mask=src_key_padding_mask
                )[0]
                src2 = torch.cat((src2,src_treat2),-1)
        else: 
            if self.treat_attention_type == "mlp":
                src_treat2 = self.mlp_head(src_treat, src_mask_treat)
                src2  = src2 + src_treat2
        
        src = self.residual1(src, self.dropout1(src2))
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.residual2(src, self.dropout2(src2))
        src = self.norm2(src)
        return src


class TransformerModel(nn.Module):
    def __init__(self, input_long_size, hidden_dim, baseline_size, trans_layers, reconstruction_size, planned_treatments, max_pos_encoding, d_pos_embed, nheads, ff_dim, output_dims, treatment_dim = 0, **kwargs):
        super().__init__()
        
        self.transformer_layers = trans_layers
      
        self.PositionalEncoding = PositionalEncoding(1,max_pos_encoding,d_pos_embed)

        self.emission_net = torch.nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,reconstruction_size))

        
        self.treatment_heads = False
        
        #self.T_cond = T_cond

        #self.horizon = T_horizon
        self.reconstruction_size = reconstruction_size

        self.planned_treatments = planned_treatments

        self.hidden_dim = hidden_dim
        #self.input_long_size = input_long_size
        self.use_statics = False

        t_embed = 8
        d_in = input_long_size + d_pos_embed + t_embed
        if self.use_statics:
            base_hidden_dim = 8
            self.embed_statics = nn.Linear(baseline_size,base_hidden_dim)
            d_in += base_hidden_dim

        self.input_mapping = nn.Linear(d_in, hidden_dim)

        self.expert = False

        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim,nheads,ff_dim, norm = "layer", nhead_treat = 0, treat_attention_type = None, linear_type = "normal") for n in range(trans_layers)])
        self.output_layer = nn.Linear(hidden_dim,reconstruction_size)

        self.output_dims = output_dims

        self.survival_loss = False

        self.joint = False
        
        self.mlp_head = False

        self.act = lambda x: x * torch.sigmoid(x)
        self.treatment_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= t_embed),
             nn.Linear(t_embed, t_embed))
    def forward(self,X,T):
        #X_long,A_long,X_forward, A_forward, times_X, times_Y):
        Tembed = self.act(self.treatment_embedder(T[:,0].float()))[...,None].repeat(1,1,X.shape[-1])
        X_long = torch.cat((X,Tembed),1)
        Y_long = torch.zeros_like(X_long,device = X.device)
        
        times_ = torch.arange(2*X.shape[-1],device = X.device)[None,:].repeat(X.shape[0],1)
        times_X = times_[:,:X.shape[-1]]
        times_Y = times_[:,X.shape[-1]:]
        #X = self.get_long_input_x(X_long, A_long)
        #Y = self.get_long_input_y(X_forward, X_long, A_forward, A_long)
        times = self.get_times_full(times_X,times_Y)

    #def forward(self,B,X, Y, times, mask_pre, mask_post):

        #times = times * (-1) + 1
        
        pos_encodings = self.PositionalEncoding(times).permute(0,2,1)

        sequence = torch.cat((X_long,Y_long),-1)

        treatment_sequence = None
        treat_full = None

        x_full = torch.cat((pos_encodings,sequence),1) # N x D x T

        if self.use_statics:
            x_static = self.embed_statics(B)[:,:,None].repeat(1,1,x_full.shape[-1])
            x_full = torch.cat((x_full,x_static),1)
        
        x = self.input_mapping(x_full.permute(0,2,1)).permute(1,0,2)

        x_treat = None
        
        src_mask, src_mask_treat, key_padding_mask = self.get_src_mask(X_long,Y_long)
      
        for layer in self.layers:
            x = layer(x, x_treat, src_mask = src_mask, src_mask_treat = src_mask_treat, src_key_padding_mask = key_padding_mask)
        x_out = self.output_layer(x)
        
        forecast = x_out.permute(1,2,0)
        forecast_X = forecast[...,:X_long.shape[-1]-1]
        forecast_Y = forecast[...,X_long.shape[-1]-1:-1]
        return forecast_Y

    def get_long_input_x(self, X_long, M_long, A_long):
        
        return torch.cat((X_long, A_long),1), None

    def get_long_input_y(self, X_forward, X_long, A_forward, A_long, mask_post):
        
        long_input_y = torch.cat((torch.zeros(X_forward.shape, device = X_long.device),A_forward),1), None
        return long_input_y

    def get_times_full(self,times_X, times_Y):
        if times_Y is not None:
            times_full = torch.cat((times_X,times_Y),-1)
        else:
            times_full = times_X
        return times_full


    def get_src_mask(self,X,Y):
        """
        Computes the attention mask for the vitals and the treatment heads.
        """
        tx = X.shape[-1]
        ty = Y.shape[-1]

        src_mask_x = torch.triu(torch.ones(tx,tx)) # Tx x Tx
        src_mask_x = torch.cat((src_mask_x,torch.zeros(ty,tx)),0) # (Tx + Ty) x Tx

        src_mask_y = torch.ones(tx+ty,ty)
        src_mask = torch.cat((src_mask_x,src_mask_y),1)
        src_mask = ~ (src_mask.bool().to(X.device)).T

        src_mask_treat = torch.triu(torch.ones(tx+ty,tx+ty))
        src_mask_treat = ~(src_mask_treat.bool().to(X.device)).T
        
        key_padding_mask = None
        
        return src_mask , src_mask_treat, key_padding_mask

        


class Transformer(pl.LightningModule):
    def __init__(self, input_long_size, hidden_dim, baseline_size, lr, trans_layers, weight_decay, T_cond, T_horizon, reconstruction_size, planned_treatments, max_pos_encoding, d_pos_embed, nheads, ff_dim, output_dims, treatment_dim = 0, dropout_p = 0,  nhead_treat = 0, survival_loss = 0, treat_attention_type = None,  linear_type = "normal", **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.transformer_layers = trans_layers
      
        self.PositionalEncoding = PositionalEncoding(1,max_pos_encoding,d_pos_embed)

        self.emission_net = torch.nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,reconstruction_size))

        self.treat_attention_type = treat_attention_type
        
        if nhead_treat>0:
            self.treatment_heads = True
        else:
            if (self.treat_attention_type == "linear") or (self.treat_attention_type == "mlp"): # no real treatment heads but use treatment information
                self.treatment_heads = True
            else:
                self.treatment_heads = False
        
        self.lr = lr
        self.weight_decay = weight_decay

        self.T_cond = T_cond

        self.horizon = T_horizon
        self.reconstruction_size = reconstruction_size

        self.planned_treatments = planned_treatments

        self.hidden_dim = hidden_dim
        if self.treatment_heads:
            input_long_size = input_long_size - treatment_dim
        self.input_long_size = input_long_size
        self.use_statics = True

        d_in = input_long_size + d_pos_embed
        if self.use_statics:
            base_hidden_dim = 8
            self.embed_statics = nn.Linear(baseline_size,base_hidden_dim)
            d_in += base_hidden_dim

        self.input_mapping = nn.Linear(d_in, hidden_dim)
        if self.treatment_heads:
            self.treat_mapping = nn.Linear(treatment_dim + d_pos_embed,hidden_dim)

        if self.treat_attention_type == "expert":
            self.expert = True
        else:
            self.expert = False

        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim,nheads-nhead_treat,ff_dim,dropout_p, norm = "layer", nhead_treat = nhead_treat, treat_attention_type = self.treat_attention_type, linear_type = linear_type) for n in range(trans_layers)])
        self.output_layer = nn.Linear(hidden_dim,reconstruction_size)

        self.output_dims = output_dims

        self.survival_loss = survival_loss
        if self.survival_loss:
            self.survival_layer = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.Tanh(), nn.Linear(hidden_dim,1))

        if kwargs["data_type"] == "MMSynthetic2":
            self.joint = True # Using joint X and Y in a same vector and playing with the  mask only.
        elif kwargs["data_type"] == "MMSynthetic3":
            self.joint = True
        else:
            self.joint = False
        
        if self.treat_attention_type == "mlp":
            self.mlp_head = True
        else:
            self.mlp_head = False


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    
    def forward(self,B,X, Y, times, mask_pre, mask_post):

        #times = times * (-1) + 1
        
        pos_encodings = self.PositionalEncoding(times).permute(0,2,1)

        if self.joint:
            sequence = nn.utils.rnn.pad_sequence([torch.cat((X[0][i],Y[0][i]),-1) for i in range(len(X[0]))], batch_first = True)
            sequence = torch.cat((sequence, torch.zeros(sequence.shape[:2] + (mask_pre.shape[-1]-sequence.shape[-1],),device = X[0][0].device)),-1)
        else:
            sequence = torch.cat((X[0],Y[0]),-1)

        if X[1] is None: #classical head
            treatment_sequence = None
            treat_full = None
        else:
            if self.joint:
                treatment_sequence = nn.utils.rnn.pad_sequence([torch.cat((X[1][i],Y[1][i]),-1) for i in range(len(X[1]))], batch_first = True)
                treatment_sequence = torch.cat((treatment_sequence, torch.zeros(treatment_sequence.shape[:2] + (mask_pre.shape[-1]-treatment_sequence.shape[-1],),device = X[1][0].device)),-1)
            else:
                treatment_sequence = torch.cat((X[1],Y[1]),-1)
            if self.expert:
                treat_full = treatment_sequence
            else:
                if self.mlp_head:
                    treat_full = treatment_sequence
                else:
                    treat_full = torch.cat((pos_encodings,treatment_sequence),1)

        x_full = torch.cat((pos_encodings,sequence),1) # N x D x T

        if self.use_statics:
            x_static = self.embed_statics(B)[:,:,None].repeat(1,1,x_full.shape[-1])
            x_full = torch.cat((x_full,x_static),1)
        
        x = self.input_mapping(x_full.permute(0,2,1)).permute(1,0,2)

        if treat_full is not None:
            if self.expert:
                x_treat = treat_full
            else:
                if self.mlp_head:
                    x_treat = treat_full.permute(2,0,1)
                else:
                    x_treat = self.treat_mapping(treat_full.permute(0,2,1)).permute(1,0,2)
        else:
            x_treat = None
        
        src_mask, src_mask_treat, key_padding_mask = self.get_src_mask(X,Y, mask_pre, mask_post)
      
        for layer in self.layers:
            x = layer(x, x_treat, src_mask = src_mask, src_mask_treat = src_mask_treat, src_key_padding_mask = key_padding_mask)
        x_out = self.output_layer(x)
        
        if self.survival_loss:
            x_out_surv = self.survival_layer(x).permute(1,2,0)
        else:
            x_out_surv = None

        return x_out.permute(1,2,0), x_out_surv

    def get_individual_src_mask(self,mask_pre,mask_post):
        tx = mask_pre.sum()
        n_lead = torch.where(mask_pre)[0][0].item()
        n_between = torch.where(mask_post)[0][0].item()-torch.where(mask_pre)[0][-1].item()-1
        ty = mask_post.sum()
        n_tail = len(mask_post) - torch.where(mask_post)[0][-1].item() - 1

        src_mask_x = torch.triu(torch.ones(tx,tx)) # Tx x Tx
        before_mask_x = torch.cat((torch.zeros((n_lead+tx,n_lead)),torch.cat((torch.zeros((n_lead,tx)),src_mask_x),0)),1)
        between_mask_x = torch.cat((torch.cat((before_mask_x,torch.zeros((n_lead+tx,n_between))),-1), torch.zeros((n_lead, n_lead+tx + n_between))),0)
        
        src_mask_y = torch.cat((between_mask_x,torch.zeros(ty,n_lead + tx + n_between)),0)
        temp_y = torch.cat((torch.zeros((n_lead,ty)),torch.ones((tx,ty)),torch.zeros((n_between,ty)),torch.ones(ty,ty)),0)
        src_mask_y = torch.cat((src_mask_y,temp_y),1)

        src_mask_full_ = torch.cat((src_mask_y,torch.zeros((n_lead + tx + n_between + ty, n_tail))),1)
        src_mask_full = torch.cat((src_mask_full_,torch.zeros((n_tail,n_lead + tx + n_between + ty + n_tail))),0)
        
        src_mask_full = ~(src_mask_full.bool().to(mask_pre.device)).T
        
        return src_mask_full.fill_diagonal_(False) #ensures self attention for each element

    def get_individual_trt_mask(self,mask_pre,mask_post):
        tx = mask_pre.sum()
        n_lead = torch.where(mask_pre)[0][0].item()
        n_between = torch.where(mask_post)[0][0].item()-torch.where(mask_pre)[0][-1].item()-1
        ty = mask_post.sum()
        n_tail = len(mask_post) - torch.where(mask_post)[0][-1].item() - 1

        src_mask_x = torch.triu(torch.ones(tx,tx)) # Tx x Tx
        before_mask_x = torch.cat((torch.zeros((n_lead+tx,n_lead)),torch.cat((torch.zeros((n_lead,tx)),src_mask_x),0)),1)
        between_mask_x = torch.cat((torch.cat((before_mask_x,torch.zeros((n_lead+tx,n_between))),-1), torch.zeros((n_lead, n_lead+tx + n_between))),0)
        
        src_mask_y = torch.cat((between_mask_x,torch.zeros(ty,n_lead + tx + n_between)),0)
        temp_y = torch.cat((torch.zeros((n_lead,ty)),torch.ones((tx,ty)),torch.zeros((n_between,ty)),torch.triu(torch.ones(ty,ty))),0)
        src_mask_y = torch.cat((src_mask_y,temp_y),1)

        src_mask_full_ = torch.cat((src_mask_y,torch.zeros((n_lead + tx + n_between + ty, n_tail))),1)
        src_mask_full = torch.cat((src_mask_full_,torch.zeros((n_tail,n_lead + tx + n_between + ty + n_tail))),0)
        
        src_mask_treat = ~(src_mask_full.bool().to(mask_pre.device)).T

        return src_mask_treat.fill_diagonal_(False) #ensures self attention for each element


    def get_src_mask(self,X,Y, mask_pre, mask_post):
        """
        Computes the attention mask for the vitals and the treatment heads.
        """
        if self.joint:
            src_mask = torch.stack([ self.get_individual_src_mask(mask_pre[i], mask_post[i]) for i in range(mask_pre.shape[0])])
            src_mask_treat = torch.stack([ self.get_individual_trt_mask(mask_pre[i], mask_post[i]) for i in range(mask_pre.shape[0])])
            key_padding_mask = ~(mask_pre | mask_post) # this resulted in nans in the loss
            key_padding_mask = None
        else:    
            tx = X[0].shape[-1]
            ty = Y[0].shape[-1]

            src_mask_x = torch.triu(torch.ones(tx,tx)) # Tx x Tx
            src_mask_x = torch.cat((src_mask_x,torch.zeros(ty,tx)),0) # (Tx + Ty) x Tx

            src_mask_y = torch.ones(tx+ty,ty)
            src_mask = torch.cat((src_mask_x,src_mask_y),1)
            src_mask = ~ (src_mask.bool().to(X[0].device)).T

            src_mask_treat = torch.triu(torch.ones(tx+ty,tx+ty))
            src_mask_treat = ~(src_mask_treat.bool().to(X[0].device)).T
            
            key_padding_mask = None
        
        return src_mask , src_mask_treat, key_padding_mask

    def get_long_input_x(self, X_long, M_long, A_long, mask_pre):
        
        if self.treatment_heads:
            if self.joint:
                return  [torch.cat((X_long[i,:,mask_pre[i]],M_long[i,:,mask_pre[i]]),0) for i in range(len(X_long))], [A_long[i,:,mask_pre[i]] for i in range(len(X_long))]
            else:
                return torch.cat((X_long,M_long),1), A_long
        else:
            if self.joint:
                return [torch.cat((X_long[i,:,mask_pre[i]],M_long[i,:,mask_pre[i]],A_long[i,:,mask_pre[i]]),0) for i in range(len(X_long))], None
            else:
                return torch.cat((X_long,M_long,A_long),1), None

    def get_long_input_y(self, X_forward, X_long, M_forward, A_forward, M_long, A_long, mask_post):
        
        if self.planned_treatments:
            if self.treatment_heads:
                if self.joint:
                    long_input_y = [torch.zeros((X_long.shape[1]+M_long.shape[1],mask_post[i].sum()), device = X_long.device) for i in range(len(mask_post))], [A_long[i,:,mask_post[i]] for i in range(len(X_long))]
                else:
                    long_input_y = torch.cat((torch.zeros(X_forward.shape, device = X_long.device),torch.zeros(M_forward.shape, device = X_long.device)),1), A_forward
            else:
                if self.joint:
                    long_input_y = [torch.cat((torch.zeros((X_long.shape[1]+M_long.shape[1],mask_post[i].sum()), device = X_long.device),A_long[i,:,mask_post[i]]),0) for i in range(len(mask_post))], None
                else:
                    long_input_y = torch.cat((torch.zeros(X_forward.shape, device = X_long.device),torch.zeros(M_forward.shape, device = X_long.device),A_forward),1), None
        else:
            if self.treatment_heads:
                if self.joint:
                    long_input_y =  [torch.zeros((X_long.shape[1]+M_long.shape[1],mask_post[i].sum()), device = X_long.device) for i in range(len(mask_post))], [ torch.zeros((A_long.shape[1],len(mask_post[i])), device = X_long.device) for i in range(len(X_long))]
                else:
                    long_input_y = torch.zeros(X_forward.shape[0],X_forward.shape[1]+M_forward.shape[1],X_forward.shape[2],device = X_long.device), torch.zeros(X_forward.shape[0],A_forward.shape[1],X_forward.shape[2],device = X_long.device)
            else:
                if self.joint:
                    long_input_y =  [torch.zeros((X_long.shape[1]+M_long.shape[1],mask_post[i].sum()), device = X_long.device) for i in range(len(mask_post))], None
                else:
                    long_input_y = torch.zeros(X_forward.shape[0],X_forward.shape[1]+M_forward.shape[1]+A_forward.shape[1],X_forward.shape[2],device = X_long.device), None
        return long_input_y

    def training_step(self, batch, batch_idx):

        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE, mask_pre, mask_post = self.parse_batch(batch)
        long_input_x = self.get_long_input_x(X_long, M_long, A_long, mask_pre)
        long_input_y = self.get_long_input_y(X_forward, X_long, M_forward, A_forward, M_long, A_long, mask_post)
        times_full = self.get_times_full(times_X,times_Y)

        forecast, forecast_surv = self.forward(B0,long_input_x, long_input_y, times_full, mask_pre, mask_post)
        loss_X, loss_Y, loss_surv, forecast_X, forecast_Y = self.compute_losses(forecast = forecast, forecast_surv = forecast_surv, X_long = X_long, M_long = M_long, X_forward = X_forward, M_forward = M_forward, Y_countdown = Y_countdown, CE = CE, mask_pre = mask_pre, mask_post = mask_post)

        #forecast_X = forecast[:,:,:X_long.shape[-1]]
        #forecast_Y = forecast[:,self.output_dims,X_long.shape[-1]:]
        
        #loss_X = self.compute_mse(forecast_X,X_long,M_long)
        #loss_Y = self.compute_mse(forecast_Y,X_forward[:,self.output_dims],M_forward[:,self.output_dims])

        loss = loss_X + loss_Y + loss_surv
        self.log("train_loss_full", loss,on_epoch = True)

        self.log("train_loss_X",loss_X, on_epoch = True)
        self.log("train_loss_Y",loss_Y, on_epoch = True)
        self.log("train_loss_surv",loss_surv, on_epoch = True)

        return {"loss": loss}
       
    def compute_mse(self,pred,target,mask):

        if isinstance(pred,list):
            mse = torch.stack([((pred[i]-target[i]).pow(2)*mask[i]).sum() for i in range(len(pred))]).sum() / torch.stack([m.sum() for m in mask]).sum()
            return mse
        else:
            if mask.sum()==0:
                # This is to deal with batches with no samples in them
                return 0 * pred.sum()
            return ((pred-target).pow(2)*mask).sum()/mask.sum()

    def parse_batch(self,batch):
        if self.joint:
            X,A, M, B, mask_pre, mask_post, times = batch
            return B, X, M, A, None, None, None, times, None, None, None, mask_pre, mask_post
        else:
            X,Y,T,Y_cf,p, B, Ax, Ay, Mx, My, times_X, times_Y, Y_countdown, CE = batch
            #A_future = torch.cat((Ax[...,-1][...,None],Ay),-1)
            A_future = Ay
            return B, X, Mx, Ax, Y, My, A_future, times_X, times_Y, Y_countdown, CE, None, None

    def compute_surv_loss(self, forecast_surv, Y_countdown, CE):
        if forecast_surv is None:
            return 0
        else:
            #using x10 to have something closer to 0
            mse = (Y_countdown-10*forecast_surv[:,0]).pow(2).mean(1)
            mse_ce = torch.nn.functional.relu(Y_countdown-10*forecast_surv[:,0]).pow(2).mean(1)
            loss_surv = CE[:,0]*mse_ce + (1-CE[:,0])*mse
            return loss_surv.mean()

    def compute_losses(self,forecast, forecast_surv, X_long, M_long, X_forward, M_forward, Y_countdown, CE, mask_pre, mask_post):

        if self.joint:
            # We assume no gap between mask_pre and mask_post !!!
            assert(mask_pre[:,0].sum() == mask_pre.shape[0]) # We assume that there is no leading False in the mask_pre (the time sereis starts at 0)
            mask_pre_shift = torch.cat((mask_pre[:,1:],torch.zeros((mask_pre.shape[0],1),device = mask_pre.device).bool()),1) # shifting the mask pre by 1 (beecause the last time steps predict future)
            mask_post_shift = torch.cat((mask_post[:,1:],torch.zeros((mask_post.shape[0],1),device = mask_pre.device).bool()),1)

            mask_pre_rw = torch.cat((torch.zeros((mask_pre.shape[0],1),device = mask_pre.device).bool(),(mask_pre[:,1:])),1)

            forecast_X = forecast[mask_pre_shift[:,None,:].repeat(1,X_long.shape[1],1)]
            forecast_Y = forecast[mask_post_shift[:,None,:].repeat(1,X_long.shape[1],1)]
            
            Mx = M_long[mask_pre_rw[:,None,:].repeat(1,X_long.shape[1],1)]
            My = M_long[mask_post[:,None,:].repeat(1,X_long.shape[1],1)]
            
            Xx = X_long[mask_pre_rw[:,None,:].repeat(1,X_long.shape[1],1)]
            Xy = X_long[mask_post[:,None,:].repeat(1,X_long.shape[1],1)]

            loss_X = self.compute_mse(forecast_X,Xx,Mx)
            loss_Y = self.compute_mse(forecast_Y,Xy,My)
            
            forecast_X = [forecast[i,:,mask_pre_shift[i]] for i in range(forecast.shape[0])]
            forecast_Y = [forecast[i,:,mask_post_shift[i]] for i in range(forecast.shape[0])]
            
        else:
            forecast_X = forecast[:,self.output_dims,:X_long.shape[-1]-1]
            forecast_Y = forecast[:,self.output_dims,X_long.shape[-1]-1:-1]
        
            loss_X = self.compute_mse(forecast_X,X_long[:,self.output_dims,1:],M_long[:,self.output_dims,1:])
            loss_Y = self.compute_mse(forecast_Y,X_forward[:,self.output_dims],M_forward[:,self.output_dims])

        loss_surv = self.compute_surv_loss(forecast_surv, Y_countdown, CE)

        return loss_X, loss_Y, loss_surv, forecast_X, forecast_Y

    def get_times_full(self,times_X, times_Y):
        if times_Y is not None:
            times_full = torch.cat((times_X,times_Y),-1)
        else:
            times_full = times_X
        return times_full

    def validation_step(self, batch, batch_idx):

        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE, mask_pre, mask_post = self.parse_batch(batch)

        long_input_x = self.get_long_input_x(X_long, M_long, A_long, mask_pre)
        long_input_y = self.get_long_input_y(X_forward, X_long, M_forward, A_forward, M_long, A_long, mask_post)
        times_full = self.get_times_full(times_X,times_Y)
        
        forecast, forecast_surv = self.forward(B0,long_input_x, long_input_y, times_full, mask_pre, mask_post)
        loss_X, loss_Y, loss_surv, forecast_X, forecast_Y = self.compute_losses(forecast = forecast, forecast_surv = forecast_surv, X_long = X_long, M_long = M_long, X_forward = X_forward, M_forward = M_forward, Y_countdown = Y_countdown, CE = CE, mask_pre = mask_pre, mask_post = mask_post)
        
        loss = loss_X + loss_Y + loss_surv
        self.log("val_loss_full", loss,on_epoch = True)

        self.log("val_loss_X",loss_X, on_epoch = True)
        self.log("val_loss",loss_Y, on_epoch = True)
        self.log("val_loss_surv",loss_surv, on_epoch = True)

        #return {"loss": loss}

    def test_step(self, batch, batch_idx):

        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE, mask_pre, mask_post = self.parse_batch(batch)

        long_input_x = self.get_long_input_x(X_long, M_long, A_long, mask_pre)
        long_input_y = self.get_long_input_y(X_forward, X_long, M_forward, A_forward, M_long, A_long, mask_post)
        times_full = self.get_times_full(times_X,times_Y)

        forecast, forecast_surv = self.forward(B0,long_input_x, long_input_y, times_full, mask_pre, mask_post)
        loss_X, loss_Y, loss_surv, forecast_X, forecast_Y = self.compute_losses(forecast = forecast, forecast_surv = forecast_surv, X_long = X_long, M_long = M_long, X_forward = X_forward, M_forward = M_forward, Y_countdown = Y_countdown, CE = CE, mask_pre = mask_pre, mask_post = mask_post)
        #forecast_X = forecast[:,:,:X_long.shape[-1]]
        #forecast_Y = forecast[:,self.output_dims,X_long.shape[-1]:]
        
        #loss_X = self.compute_mse(forecast_X,X_long,M_long)
        #loss_Y = self.compute_mse(forecast_Y,X_forward[:,self.output_dims],M_forward[:,self.output_dims])
       
        loss = loss_X + loss_Y + loss_surv
        self.log("test_loss", loss,on_epoch = True)

        self.log("test_loss_X",loss_X, on_epoch = True)
        self.log("test_loss_Y",loss_Y, on_epoch = True)
        self.log("test_loss_surv",loss_surv, on_epoch = True)

        return {"loss": loss, "Y_pred":forecast_Y, "Y":X_forward, "M":M_forward}


    def predict_step(self, batch, batch_idx ):

        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y, Y_countdown, CE, mask_pre, mask_post = self.parse_batch(batch)

        long_input_x = self.get_long_input_x(X_long, M_long, A_long, mask_pre)
        long_input_y = self.get_long_input_y(X_forward, X_long, M_forward, A_forward, M_long, A_long, mask_post)
        times_full = self.get_times_full(times_X,times_Y)

        forecast, forecast_surv = self.forward(B0,long_input_x, long_input_y, times_full, mask_pre, mask_post)
        loss_X, loss_Y, loss_surv, forecast_X, forecast_Y = self.compute_losses(forecast = forecast, forecast_surv = forecast_surv, X_long = X_long, M_long = M_long, X_forward = X_forward, M_forward = M_forward, Y_countdown = Y_countdown, CE = CE, mask_pre = mask_pre, mask_post = mask_post)


        #forecast_X = forecast[:,:,:X_long.shape[-1]]
        #forecast_Y = forecast[:,self.output_dims,X_long.shape[-1]:]
        
        #loss_X = self.compute_mse(forecast_X,X_long,M_long)
        #loss_Y = self.compute_mse(forecast_Y,X_forward[:,self.output_dims],M_forward[:,self.output_dims])

        loss = loss_X + loss_Y + loss_surv
        #self.log("test_loss", loss,on_epoch = True)

        #self.log("test_loss_X",loss_X, on_epoch = True)
        #self.log("test_loss_Y",loss_Y, on_epoch = True)

        if self.joint:
            bs = X_long.shape[0]
            X_forward = [X_long[i,:,mask_post[i]] for i in range(bs)]
            M_forward = [M_long[i,:,mask_post[i]] for i in range(bs)]
            X_long = [X_long[i,:,mask_pre[i]] for i in range(bs)]
            M_long = [M_long[i,:,mask_pre[i]] for i in range(bs)]

        return {"loss": loss, "Y_pred":forecast_Y, "Y":X_forward, "M": M_forward, "X": X_long, "M_x": M_long}
        

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=48)
        parser.add_argument('--trans_layers', type=int, default=2)
        parser.add_argument('--nheads', type=int, default=8, help = " total number of heads in the transformer (normal + treat heads)")
        parser.add_argument('--ff_dim', type=int, default=512, help = "hidden_units of the feed-forward in the transformer architecture")
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--dropout_p', type=float, default=0.)
        parser.add_argument('--planned_treatments', type=str2bool, default=False)
        parser.add_argument('--nhead_treat', type = int, default = 0, help = "number of treatment heads to use")
        parser.add_argument('--max_pos_encoding', type=int, default=100, help = "Maximum time (used for computing the continuous positional embeddings)")
        parser.add_argument('--d_pos_embed', type=int, default=10,help = "Dimension of the positional embedding")
        parser.add_argument('--treat_attention_type', type=str, default=None, help = "Expert or Linear")
        parser.add_argument('--linear_type', type=str, default="normal", help = "normal or all")
        return parser

