import pytorch_lightning as pl
import sys
sys.path.insert(0,"../")
from utils import DATA_DIR
import utils
from utils import str2bool
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import os
import argparse
import numpy as np
from scipy.integrate import odeint
import pandas as pd

sys.path.insert(0,"../../ief/data/")
from semi_synthetic.ss_data import *


def create_mm_semi_synthetic_data(fold, N_train):
    ddata = load_ss_data(N_train, \
                                 add_missing=True, \
                                 gen_fly=True, \
                            in_sample_dist=False)
    hparams = dict()
    hparams['dim_base']  = ddata['train']['B'].shape[-1]
    hparams['dim_data']  = ddata['train']['X'].shape[-1]
    hparams['dim_treat'] = ddata['train']['A'].shape[-1]
    

    return ddata, hparams


class SemiSyntheticMMDataModule(pl.LightningDataModule):
    def __init__(self,batch_size = 32, seed = 421, N_ts = 1000, noise_std = 0., num_workers = 4, T_cond = 0, T_horizon = 0, fold = 0, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 1
        self.output_dim = 1

        self.N_train = N_ts
        self.noise_std = noise_std

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        self.cumulative_y = False

        self.fold = fold


    def load_helper(self, ddata, tvt,  oversample=True, att_mask=False):
        fold = self.fold; batch_size = self.batch_size
    
        B  = torch.from_numpy(ddata['B'].astype('float32'))
        X  = torch.from_numpy(ddata['X'].astype('float32'))
        A  = torch.from_numpy(ddata['A'].astype('float32'))
        M  = torch.from_numpy(ddata['M'].astype('float32'))

        Y = torch.from_numpy(ddata['Y'].astype('float32'))
        CE = torch.from_numpy(ddata['CE'].astype('float32'))
        
        
        if 'digitized_y' in ddata[fold][tvt]:
            print ('using digitized y')
            Y  = torch.from_numpy(ddata[fold][tvt]['digitized_y'].astype('float32'))
            if self.cumulative_y:
                Y = torch.cumsum(Y,1)
        else:
            Y  = torch.from_numpy(ddata[fold][tvt]['ys_seq'][:,[0]]).squeeze()

        CE = torch.from_numpy(ddata[fold][tvt]['ce'].astype('float32'))

        #NORMALIZATION
        if tvt=="train":
            self.means_x = X.mean(dim=(0,1))[None,None,:]
            self.stds_x = X.std(dim=(0,1))[None,None,:]
            
            X = (X-self.means_x)/self.stds_x
        else:
            X = (X-self.means_x)/self.stds_x
        
        events = Y
        Y = X[:,self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)
        X = X[:, :self.T_cond].permute(0,2,1)
        M_after = M[:, self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)
        M_before = M[:, :self.T_cond].permute(0,2,1)
        A_x = A[:,:self.T_cond].permute(0,2,1)
        A_y = A[:,self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)

        times_X = torch.arange(self.T_cond)
        times_Y = torch.arange(self.T_cond,self.T_cond+self.T_horizon)

        T = torch.zeros(Y.shape[0])
        Y_cf = torch.zeros(Y.shape[0])
        p = torch.zeros(Y.shape[0])
        data = TensorDataset(X, Y, T, Y_cf,p,  B[:], A_x,A_y, M_before, M_after)
        
        self.conditional_dim = X.shape[1] + A_x.shape[1] + M_before.shape[1]
        self.conditional_len = X.shape[-1]
        self.output_dim = Y.shape[1]
        self.baseline_size = B.shape[-1]
        self.treatment_dim = A_x.shape[1]

        return data


    def prepare_data(self):

        self.ddata, hparams = create_mm_semi_synthetic_data(fold = self.fold,N_train = self.N_train)
        self.train_dataset = self.load_helper(self.ddata,tvt = "train")
        self.val_dataset = self.load_helper(self.ddata,tvt = "valid")
        self.test_dataset = self.load_helper(self.ddata,tvt = "test")

        self.train_batch_size = self.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size

        #self.conditional_dim = train_dataset.conditional_dim
        #self.conditional_len = train_dataset.conditional_len
        #self.output_dim = train_dataset.output_dim
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self, shuffle = False):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--N_ts', type=int, default=1000)
        parser.add_argument('--gamma', type=float, default=0)
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--continuous_treatment', type=str2bool, default=False)
        parser.add_argument('--fixed_length', type=str2bool, default=False)
        parser.add_argument('--static_x', type=str2bool, default=False, help = "If true, returns the initial theta value as X")
        parser.add_argument('--strong_confounding', type=str2bool, default=False, help = "If true, increases the confuonding byt a particular function")
        parser.add_argument('--T_cond', type=int, default=10)
        parser.add_argument('--T_horizon', type=int, default=5)
        parser.add_argument('--num_workers', type=int, default=4)
        return parser

   
if __name__=="__main__":
    dataset = SemiSyntheticMMDataModule()
    dataset.prepare_data()

    import ipdb;  ipdb.set_trace()

