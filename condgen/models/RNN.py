import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
from condgen.utils import str2bool


class RNN_seq2seq(pl.LightningModule):
    def __init__(self, input_long_size, hidden_dim, baseline_size, lr, rnn_layers, weight_decay, T_cond, T_horizon, reconstruction_size, planned_treatments, treatment_dim = 0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.rnn_layers = rnn_layers
        self.RNN_enc = torch.nn.RNN(input_size = input_long_size, hidden_size = hidden_dim, bidirectional = True, batch_first = True, num_layers = self.rnn_layers)

        if planned_treatments: 
            self.RNN_dec = torch.nn.RNN(input_size = reconstruction_size + treatment_dim, hidden_size = hidden_dim, bidirectional = False, batch_first = True, num_layers = self.rnn_layers)
        else:
            self.RNN_dec = torch.nn.RNN(input_size = reconstruction_size, hidden_size = hidden_dim, bidirectional = False, batch_first = True, num_layers = self.rnn_layers)
        
        self.h0_net = torch.nn.Sequential(nn.Linear(baseline_size,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,2*self.rnn_layers*hidden_dim))
        self.convert_hn = torch.nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,self.rnn_layers*hidden_dim))

        self.emission_net = torch.nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,reconstruction_size))

        self.lr = lr
        self.weight_decay = weight_decay

        self.T_cond = T_cond

        self.horizon = T_horizon
        self.reconstruction_size = reconstruction_size

        self.planned_treatments = planned_treatments

        if kwargs["data_type"] == "MMSynthetic2":
            self.joint = True # Using joint X and Y in a same vecetor and playing with the  mask only.
        elif kwargs["data_type"] == "MMSynthetic3":
            self.joint = True
        else:
            self.joint = False

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    
    def forward(self,B,X, A_future):
        h0 = self.h0_net(B)
        h0_ = torch.stack(torch.chunk(h0,2*self.rnn_layers,-1))
        output, hn = self.RNN_enc(X.permute(0,2,1),h0_)
        hn = hn.view(2, self.rnn_layers, B.shape[0],hn.shape[-1])
        hn = hn[:,-1,:,:].permute(1,2,0).reshape(B.shape[0],-1)
        #hn = hn.permute(1,2,0).reshape(hn.shape[1],-1)
       
        h_dec = self.convert_hn(hn)
        h_dec = torch.stack(torch.chunk(h_dec,self.rnn_layers,-1))

        emissions = [self.emission_net(h_dec[-1])[:,None,:]]
        
        if self.planned_treatments:
            #token_in = torch.cat((X[:,:self.reconstruction_size,-1][:,None,:], A_future[:,:,0][...,None].permute(0,2,1)),-1)
            token_in = torch.cat((emissions[0], A_future[:,:,0][...,None].permute(0,2,1)),-1)
        else:
            #token_in = X[:,:self.reconstruction_size,-1][:,None,:]
            token_in = emissions[0]

        for horizon_idx in range(A_future.shape[-1]-1):
            o,h_dec = self.RNN_dec(token_in, h_dec)
            token_in = self.emission_net(o)
            emissions.append(token_in)
            if self.planned_treatments:
                token_in  = torch.cat((token_in,A_future[:,:,horizon_idx+1][...,None].permute(0,2,1)),-1)
        
        emissions_tensor = torch.cat(emissions,1)
        return emissions_tensor.permute(0,2,1)

    def training_step(self, batch, batch_idx):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        forecast = self.forward(B0,long_input, A_forward)
        target = X_forward
        loss = self.compute_mse(forecast, target, M_forward) 
        self.log("train_loss", loss,on_epoch = True)
        return {"loss": loss}

    def compute_mse(self,pred,target,mask):
        return ((pred-target).pow(2)*mask).sum()/mask.sum()

    #def parse_batch(self,batch):
    #    X,Y,T,Y_cf,p, B, Ax, Ay, Mx, My, times_X, times_Y = batch
    #    A_future = torch.cat((Ax[...,-1][...,None],Ay),-1)
    #    return B, X, Mx, Ax, Y, My, A_future
   # 
    def parse_batch(self,batch):
        if self.joint:
            X,A, M, B, mask_pre, mask_post, times = batch
            Xx = X[mask_pre[:,None,:].repeat(1,X.shape[1],1)].reshape(X.shape[0],X.shape[1],-1)
            Mx = M[mask_pre[:,None,:].repeat(1,M.shape[1],1)].reshape(M.shape[0],M.shape[1],-1)
            Ax = A[mask_pre[:,None,:].repeat(1,A.shape[1],1)].reshape(A.shape[0],A.shape[1],-1)
            
            Y = X[mask_post[:,None,:].repeat(1,X.shape[1],1)].reshape(X.shape[0],X.shape[1],-1)
            My = M[mask_post[:,None,:].repeat(1,M.shape[1],1)].reshape(M.shape[0],M.shape[1],-1)
            A_future = A[mask_post[:,None,:].repeat(1,A.shape[1],1)].reshape(A.shape[0],A.shape[1],-1)

            return B, Xx, Mx, Ax, Y, My, A_future
        else:
            X,Y,T,Y_cf,p, B, Ax, Ay, Mx, My, times_X, times_Y, Y_countdown, CE = batch
            #A_future = torch.cat((Ax[...,-1][...,None],Ay),-1)
            A_future = Ay
            Y = Y[:,:,:self.horizon]
            My = My[:,:,:self.horizon]
            A_future = A_future[:,:,:self.horizon]
            return B, X, Mx, Ax, Y, My, A_future

    def validation_step(self, batch, batch_idx):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        forecast = self.forward(B0,long_input, A_forward)
        target = X_forward
        loss = self.compute_mse(forecast,target,M_forward)
        self.log("val_loss", loss,on_epoch = True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        forecast = self.forward(B0,long_input, A_forward)
        target = X_forward
        loss = self.compute_mse(forecast, target, M_forward)
        self.log("test_loss", loss,on_epoch = True)
        return {"loss": loss, "Y_pred":forecast, "Y":target, "M": M_forward}

    def predict_step(self, batch, batch_idx, _):
        B0,X_long, M_long, A_long, X_forward, M_forward, A_forward = self.parse_batch(batch)
        long_input = torch.cat((X_long,M_long,A_long),1)
        
        forecast = self.forward(B0,long_input, A_forward)
        target = X_forward
        loss = self.compute_mse(forecast, target, M_forward)
        self.log("test_loss", loss,on_epoch = True)
        return {"loss": loss, "Y_pred":forecast, "Y":target, "M": M_forward, "X": X_long, "M_x": M_long}
        

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=8)
        parser.add_argument('--rnn_layers', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--planned_treatments', type=str2bool, default=False)
        return parser

