import pytorch_lightning as pl
import sys
from condgen.utils import DATA_DIR
#from causalode.datagen import cancer_simulation
import condgen.utils as utils
from condgen.utils import str2bool
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(DATA_DIR,"ml_mmrf"))
from ml_mmrf.data import load_mmrf


def create_mm_data(fold, data_dir):
    data_dir = os.path.join(data_dir,f"cleaned_mm{fold}_2mos_pfs_ind_seed0.pkl")
    include_baseline = "all"
    include_treatment = "lines"
    ddata = load_mmrf(fold_span = [fold], \
                              data_dir  = data_dir, \
                              digitize_K = 36, \
                              digitize_method = 'uniform', \
                              restrict_markers=[], \
                              add_syn_marker=False, \
                              window='all', \
                              data_aug=False, \
                              ablation=True, \
                              feats=[include_baseline, include_treatment])
    
    hparams = dict()
    hparams['dim_base']  = ddata[fold]['train']['b'].shape[-1]
    hparams['dim_data']  = ddata[fold]['train']['x'].shape[-1]
    hparams['dim_treat'] = ddata[fold]['train']['a'].shape[-1]

    return ddata, hparams

class MMDataModule(pl.LightningDataModule):
    def __init__(self,batch_size = 32, seed = 421, noise_std = 0., num_workers = 4, T_cond = 0, T_horizon = 0, fold = 0, data_dir = None, focus_variables = False, survival_loss = False, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 1
        self.output_dim = 1

        self.noise_std = noise_std

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        self.cumulative_y = False

        self.fold = fold
        self.data_dir = os.path.join(DATA_DIR,data_dir)

        if kwargs["model_type"] == "DMM":
            self.dmm_order = True
        else:
            self.dmm_order = False

        self.focus_variables = focus_variables
        self.survival_loss = survival_loss

    def load_helper(self, ddata, tvt,  oversample=True, att_mask=False):
        fold = self.fold; batch_size = self.batch_size
    
        B  = torch.from_numpy(ddata[fold][tvt]['b'].astype('float32'))
        X  = torch.from_numpy(ddata[fold][tvt]['x'].astype('float32'))
        A  = torch.from_numpy(ddata[fold][tvt]['a'].astype('float32'))
        M  = torch.from_numpy(ddata[fold][tvt]['m'].astype('float32'))
        y_vals   = self.ddata[fold][tvt]['ys_seq'][:,0].astype('float32')
        idx_sort = np.argsort(y_vals)
       
        CE = torch.from_numpy(ddata[fold][tvt]['ce'].astype('float32'))
        if 'digitized_y' in ddata[fold][tvt]:
            print ('using digitized y')
            Y_mm  = torch.from_numpy(ddata[fold][tvt]['digitized_y'].astype('float32'))
            if self.cumulative_y:
                Y_mm = torch.cumsum(Y_mm,1)
        else:
            Y_mm  = torch.from_numpy(ddata[fold][tvt]['ys_seq'][:,[0]]).squeeze()
        
        Y_seq = torch.from_numpy(ddata[fold][tvt]['ys_seq'][:,[0]]).squeeze()
        #Y_mm = torch.nn.functional.one_hot(torch.round(Y_seq).long(),50)
        #Y_mm = Y_mm[:,:X.shape[1]]
        
        #CE[Y_mm.sum(1)==0] = 1
        #Y_mm[Y_mm.sum(1)==0,-1] = 1
        
        #survival_mask = (Y_mm[:,:self.T_cond].sum(1) == 0) #no event in the condition window.
        if self.survival_loss:
            survival_mask = Y_seq > self.T_cond
        else:
            survival_mask = torch.ones(X.shape[0]).bool()

        Y_countdown = Y_seq[:,None] - torch.arange(X.shape[1])[None,:]

        #NORMALIZATION
        if tvt=="train":
            self.means_x = X.mean(dim=(0,1))[None,None,:]
            self.stds_x = X.std(dim=(0,1))[None,None,:]
            
            X = (X-self.means_x)/self.stds_x
        else:
            X = (X-self.means_x)/self.stds_x
        
        #if att_mask: 
        #    attn_shape  = (A.shape[0],A.shape[1],A.shape[1])
        #    Am   = get_attn_mask(attn_shape, self.ddata[fold][tvt]['a'].astype('float32'), device)
        #    data = TensorDataset(B[idx_sort], X[idx_sort], A[idx_sort], M[idx_sort], Y[idx_sort], CE[idx_sort], Am[idx_sort])
        #else: 
        events = Y_mm[idx_sort,:self.T_cond+self.T_horizon]
        Y = X[idx_sort,self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)
        X = X[idx_sort, :self.T_cond].permute(0,2,1)
        M_after = M[idx_sort, self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)
        M_before = M[idx_sort, :self.T_cond].permute(0,2,1)
        
        Y_countdown = Y_countdown[idx_sort,:self.T_cond+self.T_horizon]
        CE = CE[idx_sort]
        survival_mask = survival_mask[idx_sort]

        #point_change = torch.cat((torch.zeros(A.shape[0],1),(A[:,1:,1]-A[:,:-1,1])),-1) # just one when the treatment changes and 0 otherwise
        #A = torch.cat((A,point_change[...,None]),-1)
        
        A_x = A[idx_sort,:self.T_cond].permute(0,2,1)
        A_y = A[idx_sort,self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)

        times_X = torch.arange(self.T_cond)[None,:].repeat(Y.shape[0],1)
        times_Y = torch.arange(self.T_cond,self.T_cond+self.T_horizon)[None,:].repeat(Y.shape[0],1)

        T = torch.zeros(Y.shape[0])
        Y_cf = torch.zeros(Y.shape[0])
        p = torch.zeros(Y.shape[0])

        if self.dmm_order:
            X = torch.cat((X,Y),-1).permute(0,2,1)
            A = torch.cat((A_x,A_y),-1).permute(0,2,1)
            M = torch.cat((M_before,M_after),-1).permute(0,2,1)

            self.conditional_dim = X.shape[2] + A.shape[2] + M_before.shape[2]
            self.conditional_len = X.shape[1]
            self.output_dim = Y.shape[2]
            self.baseline_size = B.shape[-1]
            self.treatment_dim = A.shape[2]
            self.ts_dim = X.shape[2]

            data = TensorDataset(B[survival_mask], X[survival_mask], A[survival_mask], M[survival_mask] , events[survival_mask], CE[idx_sort][survival_mask]) 
        
        else:

            data = TensorDataset(X[survival_mask], Y[survival_mask], T[survival_mask], Y_cf[survival_mask],p[survival_mask],  B[idx_sort][survival_mask], A_x[survival_mask],A_y[survival_mask], M_before[survival_mask], M_after[survival_mask], times_X[survival_mask], times_Y[survival_mask], Y_countdown[survival_mask], CE[survival_mask])
        
            self.conditional_dim = X.shape[1] + A_x.shape[1] + M_before.shape[1]
            self.conditional_len = X.shape[-1]
            self.output_dim = Y.shape[1]
            self.baseline_size = B.shape[-1]
            self.treatment_dim = A_x.shape[1]
            self.ts_dim = X.shape[1]

        if self.focus_variables:
            self.output_dims = [7,8,12,13,15]
            #self.output_dim = len(self.output_dims)
        else:
            self.output_dims = [i for i in range(self.output_dim)]
        return data


    def prepare_data(self):
        
        
        self.ddata, hparams = create_mm_data(fold = self.fold, data_dir = self.data_dir)
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
        parser.add_argument('--T_cond', type=int, default=10)
        parser.add_argument('--T_horizon', type=int, default=5)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--data_dir', type=str, default="ia15/processed/")
        parser.add_argument('--focus_variables', type=str2bool, default=False, help = "if true, only trains on the main variables (but use everything as input)")
        return parser

   
if __name__=="__main__":
    dataset = MMDataModule()
    dataset.prepare_data()

    import ipdb;  ipdb.set_trace()
    #dataset = Dataset(path="/home/edward/Data/pre_processed_mimic/models/")
    #datam = FluidDataModule(path = "/home/edward/Data/pre_processed_mimic/models/")

    #datam.prepare_data()

    #for i,b in enumerate(datam.train_dataloader()):
    #    print(b)
    #    import ipdb; ipdb.set_trace()
    #    #break

    #dataset = CancerDataModule(batch_size = 32, seed = 23, chemo_coeff = 2, radio_coeff = 2, window_size = 15, num_time_steps = 20, t_limit = 10, num_workers = 4, )
    #dataset.prepare_data()
