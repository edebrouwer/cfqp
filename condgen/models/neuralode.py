import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
from condgen.utils import str2bool
from torchdiffeq import odeint

class ContinuousTreatmentODE(nn.Module):
    def __init__(self,h_dim,u_dim,shared_u_dim, continuous_treatment, fun_treatment, dropout_p, planned_treatments):
        """
        h_dim : dimension of the hidden for the general dynamics
        u_dim : dimension of the hidden for the treatment dynamics
        shared_u_dim : how many dimension of u are used to impact h
        """
        super().__init__()

        self.decoder = MLPSimple(input_dim = h_dim+1, output_dim = h_dim, hidden_dim = 4*h_dim, depth = 4, activations = [nn.ReLU() for _ in range(4)])
        self.treatment_fun = MLPSimple(input_dim = 1, output_dim = 1, hidden_dim = u_dim, depth = 4, activations = [nn.ReLU() for _ in range(4)] )
        
        self.h_dim = h_dim
        self.u_dim = u_dim

        self.continuous_treatment = continuous_treatment
        self.fun_treatment = fun_treatment

        self.planned_treatments = planned_treatments

    def forward_ode(self,h,t):
        h_gen = h[...,:self.h_dim_gen]
        u = h[...,self.u_dim:]
        h_ = self.h_fun(h_gen)
        u_ = self.u_fun(u)
        return torch.cat((h_,u_),-1)

    def forward(self,h,t, A_forward, times):
        if self.planned_treatments:
            treat_times = (A_forward * times)
            time_diff = t-treat_times
            treatment_impact = self.treatment_fun(time_diff.permute(0,2,1)).permute(0,2,1)
            treatment_impact_selected = treatment_impact * (treat_times >0) * (time_diff>0)
            treatment_impact = treatment_impact_selected.sum(-1)
        else:
            treatment_impact = torch.zeros(h.shape[0],1,device=h.device)
        x_ = torch.cat((h,treatment_impact),1)
        x_out = self.decoder(x_)
        return x_out    


class MLPSimple(nn.Module):
    def __init__(self,input_dim,output_dim, hidden_dim, depth, activations = None, dropout_p = None):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim,output_dim))
        if activations is None:
            activations = [nn.ReLU() for _ in range(depth)]
        if dropout_p is None:
            dropout_p = [0. for _ in range(depth)]
        assert len(activations) == depth
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Dropout(dropout_p[i]),activations[i]) for i in range(depth)])
    def forward(self,x):
        x = self.input_layer(x)
        for mod in self.layers:
            x = mod(x)
        x = self.output_layer(x)
        return x

class NeuralODE(pl.LightningModule):
    def __init__(self, input_long_size, hidden_dim, baseline_size, lr, rnn_layers, weight_decay, T_cond, T_horizon, reconstruction_size, planned_treatments, treatment_dim = 0, dropout_p = 0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.rnn_layers = rnn_layers
        self.RNN_enc = torch.nn.RNN(input_size = input_long_size, hidden_size = hidden_dim, bidirectional = True, batch_first = True, num_layers = self.rnn_layers)

        self.decoder = ContinuousTreatmentODE(hidden_dim,hidden_dim,1,continuous_treatment = True, fun_treatment = True, dropout_p = dropout_p, planned_treatments = planned_treatments) 


        self.h0_net = torch.nn.Sequential(nn.Linear(baseline_size,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,2*self.rnn_layers*hidden_dim))
        self.convert_hn = torch.nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,hidden_dim))

        self.emission_net = torch.nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,reconstruction_size))

        self.lr = lr
        self.weight_decay = weight_decay

        self.T_cond = T_cond

        self.horizon = T_horizon
        self.reconstruction_size = reconstruction_size

        self.planned_treatments = planned_treatments

        self.hidden_dim = hidden_dim
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    
    #def ode_fun(self,t,x, A_future):
    #    return lambda t,x: self.decoder(x,t,A_future)

    def forward_ODE(self,encoding, times, A_future):
        if len(times.unique()) != times.shape[1]:
            raise ValueError("Times are not unique, not implemented !")
        else:
            times = times[0].float()
        if A_future.shape[1] >1:
            A_future = A_future[:,-1,:][:,None,:]
        
        ode_fun = lambda t,x : self.decoder(x,t,A_future,times)
        out = odeint(ode_fun, encoding, times, options = {"step_size":0.1},method = "rk4")
        return out[1:,:,:self.hidden_dim].permute(1,0,2) #Times x batch x dim

    def forward(self,B,X, A_future, times_Y):
        h0 = self.h0_net(B)
        h0_ = torch.stack(torch.chunk(h0,2*self.rnn_layers,-1))
        output, hn = self.RNN_enc(X.permute(0,2,1),h0_)
        hn = hn.view(2, self.rnn_layers, B.shape[0],hn.shape[-1])
        hn = hn[:,-1,:,:].permute(1,2,0).reshape(B.shape[0],-1)
        #hn = hn.permute(1,2,0).reshape(hn.shape[1],-1)
        hn = self.convert_hn(hn)
        hidden_ode = self.forward_ODE(hn,times_Y, A_future)
        out_x = self.emission_net(hidden_ode) # batch_size x time x dim Num time points (horizon)
        return out_x.permute(0,2,1) 

    def training_step(self, batch, batch_idx):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        times_forward = torch.cat((times_X[:,-1][:,None],times_Y),-1)
        forecast = self.forward(B0,long_input, A_forward, times_forward)
        target = X_forward
        loss = self.compute_mse(forecast, target, M_forward) 
        self.log("train_loss", loss,on_epoch = True)
        return {"loss": loss}

    def compute_mse(self,pred,target,mask):
        return ((pred-target).pow(2)*mask).sum()/mask.sum()

    def parse_batch(self,batch):
        X,Y,T,Y_cf,p, B, Ax, Ay, Mx, My, times_X, times_Y = batch
        A_future = torch.cat((Ax[...,-1][...,None],Ay),-1)
        return B, X, Mx, Ax, Y, My, A_future, times_X, times_Y

    def validation_step(self, batch, batch_idx):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)

        times_forward = torch.cat((times_X[:,-1][:,None],times_Y),-1)
        forecast = self.forward(B0,long_input, A_forward, times_forward)
        target = X_forward
        loss = self.compute_mse(forecast,target,M_forward)
        self.log("val_loss", loss,on_epoch = True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        times_forward = torch.cat((times_X[:,-1][:,None],times_Y),-1)
        forecast = self.forward(B0,long_input, A_forward, times_forward)
        target = X_forward
        loss = self.compute_mse(forecast, target, M_forward)
        self.log("test_loss", loss,on_epoch = True)
        return {"loss": loss, "Y_pred":forecast, "Y":target, "M": M_forward}

    def predict_step(self, batch, batch_idx, _):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward, times_X, times_Y = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        times_forward = torch.cat((times_X[:,-1][:,None],times_Y),-1)
        forecast = self.forward(B0,long_input, A_forward, times_forward)
        target = X_forward
        loss = self.compute_mse(forecast, target, M_forward)

        return {"loss": loss, "Y_pred":forecast, "Y":target, "M": M_forward, "X": X_long, "M_x": M_long}
        

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=8)
        parser.add_argument('--rnn_layers', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--dropout_p', type=float, default=0.)
        parser.add_argument('--planned_treatments', type=str2bool, default=False)
        return parser

