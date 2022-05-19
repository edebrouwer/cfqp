import pytorch_lightning as pl
import torch
import torch.nn as nn
import functools
import numpy as np
from distutils.util import strtobool
from torch.optim.lr_scheduler import ReduceLROnPlateau
import condgen.models.samplers as samplers
from condgen.models.CFGAN import ImageEmbedder, GaussianFourierProjection, Dense
from pyro.distributions import Normal
from condgen.models.scorers import ConditionalEmbedder
import torch.nn.functional as F

def create_checkerboard_mask(h, w, invert=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask

def create_channel_mask(c_in, invert=False):
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1 - mask
    return mask

class SqueezeFlow(nn.Module):

    def forward(self, z, ldj, reverse=False, X  = None, T = None):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C//4, H*2, W*2)
        return z, ldj

class SplitFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj, reverse=False, X = None, T = None):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1,2,3])
        else:
            z_split = self.prior.sample(sample_shape=z.shape).to(z.device)
            z = torch.cat([z, z_split], dim=1)
            ldj -= self.prior.log_prob(z_split).sum(dim=[1,2,3])
        return z, ldj

class CouplingLayer(nn.Module):

    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)

    def forward(self, z, ldj, reverse=False, X=None, T = None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if X is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(z_in, X = X, T = T)
            #nn_out = self.network(torch.cat([z_in, orig_img], dim=1))
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1,2,3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1,2,3])

        return z, ldj

class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2*c_hidden, 2*c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3, c_cond = 1, one_D = False):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [nn.Sequential(GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden))]
        layers += [nn.Sequential(ConcatELU(),
                   nn.Conv2d(2*c_hidden, c_out, kernel_size=3, padding=1))]
        self.nn = torch.nn.ModuleList(layers)

        self.nn[-1][-1].weight.data.zero_()
        self.nn[-1][-1].bias.data.zero_()
        
        """
        cond_layers = []
        cond_layers += [nn.Conv2d(c_cond, c_hidden, kernel_size=3, padding=1)]
        
        for layer_index in range(num_layers):
            cond_layers += [nn.Sequential(GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden))]
        cond_layers += [nn.Sequential(ConcatELU(),
                   nn.Conv2d(2*c_hidden, c_out, kernel_size=3, padding=1))]
        self.nn_cond = torch.nn.ModuleList(cond_layers)

        self.nn_cond[-1][-1].weight.data.zero_()
        self.nn_cond[-1][-1].bias.data.zero_()
        """

        embed_dim = 256 #conditional embedding
        #self.treatment_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
        #     nn.Linear(embed_dim, embed_dim))
    
        dense_layers = []
        dense_layers += [Dense(embed_dim, c_hidden, one_D)]
        for layer_index in range(num_layers):
            dense_layers += [Dense(embed_dim, c_hidden, one_D)]

        self.nn_dense = torch.nn.ModuleList(dense_layers)
        
        conditional_len = 28
        self.conditional_embed = ConditionalEmbedder(embed_dim = embed_dim, in_channels = c_cond, one_D = one_D, conditional_len = conditional_len)
        
    def forward(self, y, X, T):
        embed = self.conditional_embed(X,T) 
        h_in = y
        for layer_index in range(len(self.nn)-1):
            h_y = self.nn[layer_index](h_in)
            #h_embed = self.nn_dense[layer_index](embed)
            #h_in = h_y + h_embed
            h_in = h_y
        h_out = self.nn[-1](h_in)
        return h_out

class Dequantization(nn.Module):

    def __init__(self, alpha=1e-5, quants=256):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def forward(self, z, ldj, reverse=False, X = None, T = None):
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=0, max=self.quants-1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=[1,2,3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1,2,3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj

class VariationalDequantization(Dequantization):

    def __init__(self, var_flows, alpha=1e-5):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
        """
        super().__init__(alpha=alpha)
        self.flows = nn.ModuleList(var_flows)

    def dequant(self, z, ldj):
        z = z.to(torch.float32)
        img = (z / 255.0) * 2 - 1 # We condition the flows on x, i.e. the original image

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        deq_noise = torch.rand_like(z).detach()
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        for flow in self.flows:
            deq_noise, ldj = flow(deq_noise, ldj, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise) / 256.0
        ldj -= np.log(256.0) * np.prod(z.shape[1:])
        return z, ldj




class DeepSCM(pl.LightningModule):
    def __init__(self,weight_decay, lr, input_dim, output_dim, conditional_len, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        embed_dim = 128
        z_dim = 16
        #self.x_embedder = ImageEmbedder(one_D = self.hparams["one_D"],in_channels = input_dim, conditional_len = conditional_len, embed_dim = embed_dim)
        #self.y_embedder = ImageEmbedder(one_D = self.hparams["one_D"],in_channels = output_dim, conditional_len = conditional_len, embed_dim = embed_dim)
        #self.treatment_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
        #     nn.Linear(embed_dim, embed_dim))
        
        #self.x_embedder2 = ImageEmbedder(one_D = self.hparams["one_D"],in_channels = input_dim, conditional_len = conditional_len, embed_dim = z_dim)
        #self.treatment_embedder2 = nn.Sequential(GaussianFourierProjection(embed_dim=z_dim),
        #     nn.Linear(embed_dim, embed_dim))
        
        flow_layers = []

        #vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
        #                           mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
        #                           c_in=1) for i in range(4)]
        flow_layers += [Dequantization()]

        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=output_dim, c_hidden=32),
                                  mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
                                  c_in=output_dim) for i in range(2)]
        flow_layers += [SqueezeFlow()]
        for i in range(2):
            flow_layers += [CouplingLayer(network=GatedConvNet(c_in= output_dim * 4, c_hidden=48),
                                      mask=create_channel_mask(c_in= output_dim * 4, invert=(i%2==1)),
                                      c_in= output_dim * 4)]
        flow_layers += [SplitFlow(),
                    SqueezeFlow()]
        for i in range(4):
            flow_layers += [CouplingLayer(network=GatedConvNet(c_in=output_dim * 8, c_hidden=64),
                                      mask=create_channel_mask(c_in=output_dim * 8, invert=(i%2==1)),
                                      c_in=output_dim * 8)]

        self.flows = nn.ModuleList(flow_layers)
        
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        #self.q_model = nn.Sequential(nn.Linear(embed_dim*3,embed_dim),nn.ReLU(),nn.Linear(embed_dim, 2 * z_dim ))

        #y_transform = T.conditional_spline(1,context_dim = 3 * z_dim )

        #self.register_buffer('x_base_loc', torch.zeros([3, 28, 28], requires_grad=False))
        #self.register_buffer('x_base_scale', torch.ones([3, 28, 28], requires_grad=False))

    #def configure_optimizers(self):
    #    return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        lr_scheduler_config = {
        "scheduler": ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 20, verbose = True),
        "interval": "epoch",
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        "strict": True,
        "name": None,
    }
        return  {"optimizer": optimizer, "lr_scheduler":lr_scheduler_config}

    def forward(self,Y, X, T):
        return self._get_likelihood(Y,X,T)
    
    def encode(self,Y,X,T):
        z, ldj = Y, torch.zeros(Y.shape[0], device=self.device)
        for flow in self.flows:
            z,ldj = flow(z,ldj, reverse = False, X = X, T = T)
        return z, ldj

    def _get_likelihood(self,Y,X,T,return_ll = False):
        z, ldj = self.encode(Y,X,T)
        sample = self.sample(Y_shape = Y.shape, X = X, T = T, z_init = z)
        
        log_pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(Y.shape[1:])
        return bpd.mean() if not return_ll else log_px


    def validation_step(self,batch,batch_idx):
        X,Y, _, _, T = batch
        Y = (Y*255).to(torch.int32)
        loss = self._get_likelihood(Y,X,T)
        self.log("val_loss", loss,on_epoch = True)
        return loss

    def training_step(self,batch,batch_idx):
        X,Y, _, _, T = batch
        Y = (Y*255).to(torch.int32)
        loss = self._get_likelihood(Y,X,T)
        self.log("train_loss", loss,on_epoch = True)
        return loss

    def test_step(self,batch,batch_idx):
        X,Y, _, _, T = batch
        Y = (Y*255).to(torch.int32)
        loss = self._get_likelihood(Y,X,T)
        self.log("test_loss", loss,on_epoch = True)
        return loss

    @torch.no_grad()
    def sample(self, Y_shape, z_init=None, X = None, T = None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=Y_shape)
        else:
            z = z_init

        # Transform z to x by inverting the flows
        ldj = torch.zeros(Y_shape[0], device=X.device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True, X = X, T = T)
        return z

    def abduct(self, Y,X,T):
        return self.encode(Y,X,T)[0]

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--sigma', type=float, default=25.0)
        parser.add_argument('--conditional_score', type=strtobool, default=False)
        return parser



class TemporalScoreMatcher(pl.LightningModule):
    def __init__(self,weight_decay, lr, sigma, conditional_score, conditional_dim, output_dim, conditional_len, static_dim, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.sigma = sigma

        self.conditional = conditional_score
        
        self.marginal_prob_std_fn = functools.partial(self.marginal_prob_std, sigma=self.sigma)
        self.diffusion_coeff_fn = functools.partial(self.diffusion_coeff, sigma=self.sigma)
        self.score_model = TemporalScoreNet(marginal_prob_std=self.marginal_prob_std_fn, conditional_score = conditional_score, conditional_dim = conditional_dim, output_dim = output_dim, conditional_len = conditional_len, static_dim = static_dim)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        lr_scheduler_config = {
        "scheduler": ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 20, verbose = True),
        "interval": "epoch",
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        "strict": True,
        "name": None,
    }
        return  {"optimizer": optimizer, "lr_scheduler":lr_scheduler_config}

    def marginal_prob_std(self,t, sigma):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  

        Returns:
        The standard deviation.
        """    
        #t = torch.tensor(t, device=device)
        return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

    def diffusion_coeff(self,t, sigma):
        """Compute the diffusion coefficient of our SDE.

        Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

        Returns:
        The vector of diffusion coefficients.
        """
        return sigma**t

    def loss_fn(self,y,x,b, T_treat = None, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
        model: A PyTorch model instance that represents a 
          time-dependent score-based model.
        y: A mini-batch of training data.
        x : the data to condition on
        marginal_prob_std: A function that gives the standard deviation of 
          the perturbation kernel.
        eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(y.shape[0], device=y.device) * (1. - eps) + eps  
        z = torch.randn_like(y)
        std = self.marginal_prob_std(random_t, self.sigma)
        perturbed_y = y + z * std[:, None, None]
        score = self.score_model(perturbed_y, random_t, cond = (b,x), T_treat = T_treat)
        loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1,2)))
        return loss

    def process_condition(self,y):
        return y 

    def parse_batch(self,batch):
        X,Y,T,Y_cf,p, B, Ax, Ay, Mx, My, times_X, times_Y = batch
        A_future = torch.cat((Ax[...,-1][...,None],Ay),-1)
        return B, X, Mx, Ax, T, Y, My, A_future    
    def validation_step(self,batch,batch_idx):

        B, X, Mx, Ax,T, Y, My, A_future = self.parse_batch(batch)
        X_full = torch.cat((X,Mx,Ax),1)
        loss = self.loss_fn(Y, X_full, B, T)
        self.log("val_loss", loss,on_epoch = True)
        return loss

    def training_step(self,batch,batch_idx):
        B, X, Mx, Ax,T, Y, My, A_future = self.parse_batch(batch)
        X_full = torch.cat((X,Mx,Ax),1)
        
        loss = self.loss_fn(Y, X_full, B, T)
        self.log("train_loss", loss,on_epoch = True)
        return loss

    def test_step(self,batch,batch_idx):
        B, X, Mx, Ax,T, Y, My, A_future = self.parse_batch(batch)
        X_full = torch.cat((X,Mx,Ax),1)
        
        loss = self.loss_fn(Y, X_full, B, T)
        self.log("test_loss", loss,on_epoch = True)
        return loss

    def predict_step(self,batch,batch_idx, _):
        B, X, Mx, Ax,T, Y, My, A_future = self.parse_batch(batch)
        sample_dim = Y.shape[1:]
        sampler = samplers.pc_sampler
        
        X_full = torch.cat((X,Mx,Ax),1)
        cond = (B,X_full)
        sample_batch_size = B.shape[0]

        samples = sampler(self.score_model, self.marginal_prob_std_fn, self.diffusion_coeff_fn, sample_batch_size , cond = cond, T_treat = T, sample_dim = sample_dim)
        loss = 0
        return {"loss": loss, "Y_pred":samples, "Y":Y, "M": My, "X": X, "M_x": Mx}
    
    def sample(self, sampler, sample_batch_size, cond = None, T_treat = None):
        samples = sampler(self.score_model, self.marginal_prob_std_fn, self.diffusion_coeff_fn, sample_batch_size , cond = cond, T_treat = T_treat, sample_dim = (1,10))
        return samples

    def compute_mse(self,pred,target,mask):
        return ((pred-target).pow(2)*mask).sum()/mask.sum()
    
    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--sigma', type=float, default=25.0)
        parser.add_argument('--conditional_score', type=strtobool, default=False)
        return parser


