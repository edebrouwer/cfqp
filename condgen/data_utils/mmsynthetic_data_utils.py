
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
from scipy.integrate import odeint
import pandas as pd

from condgen.data_utils.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy



def generate_mm_synthetic2_ts(T, noise_std, counterfactual = False, treat_pair = [5,6]):

    treat_dict = {1:treat_1, 2: treat_2, 3 : treat_3, 4: treat_4, 5:treat_5, 6:treat_6}
    
    t = np.arange(T*4)/4
    fd = 2*np.random.randn() + 2 - 0.05*t - 0.005*t**2 #random intercept
    fu = 2*np.random.randn() + -1 + 0.0001*t + 0.005*t**2 #random intercept
    
    if treat_pair == [5,6]:
        fd += 0.4*np.sin(t)

    B = np.random.randn(2)

    if (B[0]>0) and (B[1]>0):
        f = [fd,fd]
        p_treat = 0.4
    elif (B[0]>0) and (B[1]<=0):
        f = [fd,fu]
        p_treat = 0.2
    elif (B[0]<=0) and (B[1]>0):
        f = [fu,fd]
        p_treat = 0.6
    elif (B[0]<=0) and (B[1]<=0):
        f = [fu, fu]
        p_treat = 0.4
    else:
        raise("ERROR")

    if counterfactual:
        p_treat = 1-p_treat
    #treatment assignments
    te = 0
    t_treats = []
    n_treats = []
    t_treat = 0
    while len(t_treats)<2:
        
        t_treat += np.random.poisson(12)
        
        if np.random.rand()>p_treat:
            te += treat_dict[treat_pair[0]](t_treat,t)
            n_treats.append(0)
        else:
            te += treat_dict[treat_pair[1]](t_treat,t)
            n_treats.append(1)
        t_treats.append(t_treat)

    noise = noise_std * np.random.randn(t.shape[0],2)
    base = np.array(f).T
    response = base + noise + te[...,None]
    
    treat_vec = np.zeros((t.shape[0],2))
    for n_treat,t_treat in zip(n_treats,t_treats):
        treat_vec[t==t_treat,n_treat] = 1
    
    return t, base, response, treat_vec, B


class MMSynthetic2Dataset(Dataset):
    def __init__(self,N, seed, T_cond, T_horizon, noise_std = 0.25, counterfactual = False, dmm_order = False, treat_pair = [5,6]):

        self.dmm_order = dmm_order
        
        np.random.seed(seed)
        
        ts = []
        bases = []
        responses = []
        treat_vecs = []
        Bs = []
        for n in range(N):
            t, base, response, treat_vec, B = generate_mm_synthetic2_ts(T_cond+T_horizon + 10, noise_std = noise_std, counterfactual = counterfactual, treat_pair = treat_pair)
            ts.append(t)
            bases.append(base)
            responses.append(response)
            treat_vecs.append(treat_vec)
            Bs.append(B)
        
        X = torch.Tensor(np.stack(responses))
        times = torch.Tensor(np.stack(ts))
        A = torch.Tensor(np.stack(treat_vecs))
        B = torch.Tensor(np.stack(Bs))

        self.X = X.permute(0,2,1)
        self.B = B
        self.A = A.permute(0,2,1)
        self.M = np.ones_like(self.X)

        self.conditional_dim = self.X.shape[1] + self.M.shape[1] + self.A.shape[1]
        self.output_dim = self.X.shape[1]
        self.conditional_len = self.X.shape[-1]
        self.baseline_size = self.B.shape[-1]
        self.treatment_dim = self.A.shape[1]
        self.ts_dim = self.X.shape[1]
        
        means_x = self.X.mean(dim=(0,2))[None,:,None]
        stds_x = self.X.std(dim=(0,2))[None,:,None]


        self.X = (self.X-means_x)/stds_x

        self.mask_pre_list = []
        self.mask_post_list = []
        
        for i in range(len(self.X)):
            self.mask_pre_list.append(times[i]<=T_cond)
            self.mask_post_list.append(((times[i]>T_cond) & (times[i]<=T_cond+T_horizon)))
        
        self.mask_pre = torch.stack(self.mask_pre_list)
        self.mask_post = torch.stack(self.mask_post_list)
        
        self.times = times
        
        self.counterfactual = counterfactual
        self.dmm_order = dmm_order

        if self.dmm_order:
            self.B = self.X_0
            self.X = torch.cat((self.X,self.Y_fact),-1).permute(0,2,1)
            self.A = torch.cat((self.A_x,self.A_y),-1).permute(0,2,1)
            self.M = torch.cat((self.M_before,self.M_after),-1).permute(0,2,1)
            self.CE = torch.zeros(self.X.shape[0])
            self.events = torch.zeros(self.X.shape[0],self.X.shape[1])

    def __getitem__(self,idx):
        if self.dmm_order:
            return self.B[idx], self.X[idx], self.A[idx], self.M[idx], self.events[idx], self.CE[idx]
        else:
            return self.X[idx], self.A[idx], self.M[idx], self.B[idx], self.mask_pre[idx], self.mask_post[idx], self.times[idx]
            #return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.X_0[idx], self.A_x[idx], self.A_y[idx], self.M_before[idx], self.M_after[idx], self.times_X[idx], self.times_Y[idx]

    def __len__(self):
        return self.X.shape[0]


def treat_1(t_treat,t):
    alpha2 = 0.6
    alpha3 = 0.6
    alpha1 = 10
    gamma = 2
    bl = 3
    b0 = -alpha1 / (1+np.exp(-alpha3*gamma)/2)
    alpha0 = (alpha1 + 2*b0 - bl) / (1 + np.exp(-alpha3*gamma)/2)
    
    lc = t-t_treat
    
    te = np.zeros_like(t)
    te_before = b0 + alpha1 / (1+ np.exp(-alpha2*(lc-gamma/2)))
    te_after = bl + alpha0 / (1+ np.exp(alpha3*(lc-3*gamma/2)))
    
    te[lc<gamma] = te_before[lc<gamma]
    te[lc>=gamma] = te_after[lc>=gamma]
    te[lc<=0] = 0
    
    return te

def treat_2(t_treat,t):
    alpha2 = 0.6
    alpha3 = 0.6
    alpha1 = 6
    gamma = 5
    bl = 1
    b0 = -alpha1 / (1+np.exp(-alpha3*gamma)/2)
    alpha0 = (alpha1 + 2*b0 - bl) / (1 + np.exp(-alpha3*gamma)/2)
    
    lc = t-t_treat
    
    te = np.zeros_like(t)
    te_before = b0 + alpha1 / (1+ np.exp(-alpha2*(lc-gamma/2)))
    te_after = bl + alpha0 / (1+ np.exp(alpha3*(lc-3*gamma/2)))
    
    te[lc<gamma] = te_before[lc<gamma]
    te[lc>=gamma] = te_after[lc>=gamma]
    te[lc<=0] = 0
    
    return te

def treat_3(t_treat,t):
    lc = t - t_treat
    treat_effect_horizon = 5
    te = 0.1 * (treat_effect_horizon-lc) * (lc<treat_effect_horizon) * np.polyval(np.poly([0,1.5,3,4.5]),lc)
    te[lc<=0] = 0
    return te

def treat_4(t_treat,t):
    lc = t - t_treat
    treat_effect_horizon = 6
    te = 0.05 * (treat_effect_horizon-lc) * (lc<treat_effect_horizon) * np.polyval(np.poly([0,3,4]),lc)
    te[lc<=0] = 0
    return te

def treat_5(t_treat,t):
    lc = t-t_treat
    treat_effect_horizon = 5
    te = 0.15 * (treat_effect_horizon-lc) * (lc<treat_effect_horizon) * np.polyval(np.poly([0,5]),lc)
    te[lc<=0] = 0
    return te

def treat_6(t_treat,t):
    lc = t-t_treat
    treat_effect_horizon = 7
    te = 0.1 * (treat_effect_horizon-lc) * (lc<treat_effect_horizon) * np.polyval(np.poly([0,3]),lc)
    te[lc<=0] = 0
    return te

class SyntheticMM2DataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts,  noise_std, num_workers = 4, T_cond = 0, T_horizon = 0, treat_pair = 0, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True


        self.input_dim = 2
        self.output_dim = 2

        self.N = N_ts
        self.noise_std = noise_std

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        if kwargs["model_type"] == "DMM":
            self.dmm_order = True
        else:
            self.dmm_order = False
        
        if treat_pair == 0:
            self.treat_pair = [3,4]
        elif treat_pair == 1:
            self.treat_pair = [5,6]
        else:
            self.treat_pair = None

    def prepare_data(self):

        dataset = MMSynthetic2Dataset(self.N, noise_std = self.noise_std, seed = self.seed, T_cond = self.T_cond, T_horizon = self.T_horizon, dmm_order = self.dmm_order, treat_pair = self.treat_pair)       
        self.dataset_cf = MMSynthetic2Dataset(self.N, noise_std = self.noise_std, seed = self.seed, T_cond = self.T_cond, T_horizon = self.T_horizon, dmm_order = self.dmm_order, counterfactual = True, treat_pair = self.treat_pair)

        train_idx = np.arange(len(dataset))[:int(0.5*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        if self.batch_size==0:
            self.train_batch_size = len(train_idx)
            self.val_batch_size = len(val_idx)
            self.test_batch_size = len(test_idx)
        else:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size

        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)

        self.conditional_dim = dataset.conditional_dim
        self.conditional_len = dataset.conditional_len
        self.output_dim = dataset.output_dim
        self.baseline_size = dataset.baseline_size
        self.treatment_dim = dataset.treatment_dim
        self.ts_dim = dataset.ts_dim

        self.output_dims = [i for i in range(self.output_dim)]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self, shuffle = False):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

    def cf_test_dataloader(self, shuffle = False):
        return DataLoader(
            self.dataset_cf,
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
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--T_cond', type=int, default=10)
        parser.add_argument('--T_horizon', type=int, default=5)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--treat_pair', type=int, default=0,help = "the type of treatment responses to use")
        return parser
   

def create_mm_synthetic_data(fold, N_train, confounding = False):

    nsamples        = {'train':N_train, 'valid':1000, 'test':200}
    print(f'training on {nsamples["train"]} samples')
    alpha_1_complex = False; per_missing = 0.; add_feat = 0; num_trt = 1
    ddata = load_synthetic_data_trt(fold_span = [fold], \
                                            nsamples = nsamples, \
                                            distractor_dims_b=4, \
                                            sigma_ys=0.7, \
                                            include_line=False, \
                                            alpha_1_complex=alpha_1_complex, \
                                            per_missing=per_missing, \
                                            add_feats=add_feat, \
                                            num_trt=num_trt, \
                                            sub=True, \
                                            confounding = confounding)
    hparams = dict()
    hparams['dim_base']  = ddata[fold]['train']['b'].shape[-1]
    hparams['dim_data']  = ddata[fold]['train']['x'].shape[-1]
    hparams['dim_treat'] = ddata[fold]['train']['a'].shape[-1]

    return ddata, hparams


class SyntheticMM3DataModule(pl.LightningDataModule):
    def __init__(self,batch_size = 32, seed = 421, N_ts = 1000, noise_std = 0., num_workers = 4, T_cond = 0, T_horizon = 0, fold = 0, confounding = False, **kwargs):
        
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
        self.confounding = confounding

        if kwargs["model_type"] == "DMM":
            self.dmm_order = True
        else:
            self.dmm_order = False

    def load_helper(self, ddata, tvt,  oversample=True, att_mask=False):
        fold = self.fold; batch_size = self.batch_size
    
        B  = torch.from_numpy(ddata[fold][tvt]['b'].astype('float32'))
        X  = torch.from_numpy(ddata[fold][tvt]['x'].astype('float32'))
        A  = torch.from_numpy(ddata[fold][tvt]['a'].astype('float32'))
        M  = torch.from_numpy(ddata[fold][tvt]['m'].astype('float32'))
        y_vals   = self.ddata[fold][tvt]['ys_seq'][:,0].astype('float32')
        idx_sort = np.argsort(y_vals)
        
        if 'digitized_y' in ddata[fold][tvt]:
            print ('using digitized y')
            Y  = torch.from_numpy(ddata[fold][tvt]['digitized_y'].astype('float32'))
            if self.cumulative_y:
                Y = torch.cumsum(Y,1)
        else:
            Y  = torch.from_numpy(ddata[fold][tvt]['ys_seq'][:,[0]]).squeeze()
            
        Y_seq  = torch.from_numpy(ddata[fold][tvt]['ys_seq'][:,[0]]).squeeze()
        Y_countdown = Y_seq[:,None] - torch.arange(X.shape[1])[None,:]
        Y_countdown = Y_countdown[idx_sort,:self.T_cond+self.T_horizon]
        CE = torch.from_numpy(ddata[fold][tvt]['ce'].astype('float32'))

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
        #events = Y[idx_sort,:self.T_cond+self.T_horizon]
        events = torch.zeros(X.shape[0],self.T_cond+self.T_horizon)
        Y = X[idx_sort,self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)
        X = X[idx_sort, :self.T_cond].permute(0,2,1)
        M_after = M[idx_sort, self.T_cond:self.T_cond+self.T_horizon].permute(0,2,1)
        M_before = M[idx_sort, :self.T_cond].permute(0,2,1)


        point_change = torch.cat((torch.zeros(A.shape[0],1),(A[:,1:,1]-A[:,:-1,1])),-1) # just one when the treatment changes and 0 otherwise
        A = torch.cat((A,point_change[...,None]),-1)
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
            data = TensorDataset(B, X, A, M , events, CE[idx_sort])

        else:
            mask_pre = torch.zeros((X.shape[0],X.shape[-1]+Y.shape[-1])).bool()
            mask_pre[:,:X.shape[-1]] = True
            
            mask_post = torch.zeros((X.shape[0],X.shape[-1]+Y.shape[-1])).bool()
            mask_post[:,-Y.shape[-1]] = True
        
            X = torch.cat((X,Y),-1)
            M = torch.cat((M_before,M_after),-1)
            A = torch.cat((A_x,A_y),-1)

            times = torch.cat((times_X,times_Y),-1)

            data = TensorDataset(X,A,M,B[idx_sort],mask_pre,mask_post,times)

            # self.X[idx], self.A[idx], self.M[idx], self.B[idx], self.mask_pre[idx], self.mask_post[idx], self.times[idx]
            #data = TensorDataset(X, Y, T, Y_cf,p,  B[idx_sort], A_x,A_y, M_before, M_after, times_X, times_Y, Y_countdown, CE)
        
            self.conditional_dim = X.shape[1] + A_x.shape[1] + M_before.shape[1]
            self.conditional_len = X.shape[-1]
            self.output_dim = Y.shape[1]
            self.baseline_size = B.shape[-1]
            self.treatment_dim = A_x.shape[1]
            self.ts_dim = X.shape[1]

        self.output_dims = [i for i in range(self.output_dim)]
        return data


    def prepare_data(self):

        self.ddata, hparams = create_mm_synthetic_data(fold = self.fold,N_train = self.N_train, confounding = self.confounding)
        cf_ddata, _ = create_mm_synthetic_data(fold = self.fold,N_train = self.N_train, confounding = False)

        self.train_dataset = self.load_helper(self.ddata,tvt = "train")
        self.val_dataset = self.load_helper(self.ddata,tvt = "valid")
        self.test_dataset = self.load_helper(self.ddata,tvt = "test")
        
        self.cf_dataset = self.load_helper(cf_ddata,tvt="test")

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

    def cf_test_dataloader(self, shuffle = False):
        return DataLoader(
            self.cf_dataset,
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
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--N_ts', type=int, default=1000)
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--continuous_treatment', type=str2bool, default=False)
        parser.add_argument('--fixed_length', type=str2bool, default=False)
        parser.add_argument('--static_x', type=str2bool, default=False, help = "If true, returns the initial theta value as X")
        parser.add_argument('--confounding', type=str2bool, default=False, help = "If true, adds confounding in the treatment times")
        parser.add_argument('--T_cond', type=int, default=10)
        parser.add_argument('--T_horizon', type=int, default=5)
        parser.add_argument('--num_workers', type=int, default=4)
        return parser
if __name__=="__main__":

    dataset = MMSynthetic2Dataset(N = 1000, seed = 421, T_cond = 15, T_horizon = 10)
    import ipdb; ipdb.set_trace()
    dataset = SyntheticMMDataModule()
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
