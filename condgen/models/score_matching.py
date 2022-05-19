import pytorch_lightning as pl
import torch
import torch.nn as nn
import functools
import numpy as np
from distutils.util import strtobool
from condgen.models.scorers import ScoreNet, TemporalScoreNet, ConditionalScoreNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import condgen.models.samplers as samplers

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


class ConditionalScoreMatcher(pl.LightningModule):
    def __init__(self,weight_decay, lr, sigma, conditional_score, input_dim, output_dim, conditional_len, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.sigma = sigma

        self.conditional = conditional_score
        
        self.marginal_prob_std_fn = functools.partial(self.marginal_prob_std, sigma=self.sigma)
        self.diffusion_coeff_fn = functools.partial(self.diffusion_coeff, sigma=self.sigma)
        self.score_model = ConditionalScoreNet(marginal_prob_std=self.marginal_prob_std_fn, input_dim = input_dim, output_dim = output_dim, one_D = self.hparams["one_D"], conditional_len = conditional_len)
        
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

    def loss_fn(self,X,Y,T,eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
        model: A PyTorch model instance that represents a 
          time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
          the perturbation kernel.
        eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(Y.shape[0], device=Y.device) * (1. - eps) + eps  
        z = torch.randn_like(Y)
        std = self.marginal_prob_std(random_t, self.sigma)
        std = std[(...,)+(None,)*(len(Y.shape)-1)] # expanding std to match dimension of Y
        perturbed_y = Y + z * std
        score = self.score_model(perturbed_y, random_t, X, T)
        loss = torch.mean(torch.sum((score * std + z)**2, dim=tuple(range(1,len(score.shape)))))
        return loss

    def process_condition(self,y):
        return y 

    def validation_step(self,batch,batch_idx):
        X,Y, _, _, T = batch
        loss = self.loss_fn(X,Y,T)
        self.log("val_loss", loss,on_epoch = True)
        return loss

    def training_step(self,batch,batch_idx):
        X,Y, _, _, T = batch
        loss = self.loss_fn(X,Y,T)
        self.log("train_loss", loss,on_epoch = True)
        return loss

    def test_step(self,batch,batch_idx):
        X,Y, _, _, T = batch
        loss = self.loss_fn(X,Y,T)
        self.log("test_loss", loss,on_epoch = True)
        return loss

    def sample(self, sampler, sample_batch_size, X_cond , T_cond, z = None):
        samples = sampler(self.score_model, self.marginal_prob_std_fn, self.diffusion_coeff_fn, sample_batch_size , X_cond = X_cond, T_cond = T_cond, z = z)
        return samples

    def abduct(self, sampler, sample_batch_size, X_cond , T_cond, X_obs):
        samples = sampler(self.score_model, self.marginal_prob_std_fn, self.diffusion_coeff_fn, sample_batch_size , X_cond = X_cond, T_cond = T_cond, reverse = True, z = X_obs)
        return samples

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--lr', type=float, default=0.0001)
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


