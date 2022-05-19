
import pytorch_lightning as pl
import sys
#sys.path.insert(0,"../")
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

def create_pendulum_data(N,gamma, noise_std, seed = 421, continuous_treatment = False, fixed_length = False, static_x = False, strong_confounding = False, linspace_theta = False, T_cond = 0, T_horizon = 0, counterfactual = False):

    np.random.seed(seed)
    g = 9.81
    if fixed_length:
        l_s = torch.ones(N)*(0.5*4 + 0.5)
    else:
        l_s = np.random.rand(N) * 4 + 0.5

    A = 10
    phi, delta = 1,1

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def df_dt(x,t, l):
        return x[1], -(g/l)*x[0]

    def dfu_dt(x,t,phi,delta):
            return (phi*x[1]*x[3]-delta*x[2]*x[3], -phi*x[2], phi*x[1], -delta*x[3])
    
    def df_dt_complete(x,t,l,phi,delta):
        return (x[1], -(g/l)*x[0]*(1+x[2])) + dfu_dt(x[2:],t,phi,delta)

    def fun_u(t):
        return 10*sigmoid(4*t-5)*(1-sigmoid(4*t-6))
    
    def df_dt_fun(x,t,l):
        return (x[1], -(g/l)*x[0]*(1+fun_u(t-10)))

    def vfun(x):
        return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2
        #return 0.2*(np.cos(10*x) * (3-x)**2)**2
    
    X = []
    Y_0 = []
    Y_1 = []
    if linspace_theta:
        thetas_0 = np.linspace(0.5,1.5)
    else:
        thetas_0 = np.random.rand(N)+0.5
    v_treatment = []

    t = np.linspace(0,T_cond+T_horizon,2*(T_cond+T_horizon)+1)
    t_ = t[t>=T_cond]
    t_x = t[t<=T_cond]
    t_y = t[t>T_cond]

    for i in range(N):
        theta_0 = thetas_0[i]
        
        y0 = np.array([theta_0,0])
        y = odeint(df_dt, y0, t, args = (l_s[i],))

        v_treatment.append( y[t==T_cond,1].item() )

        if not continuous_treatment:
            v_new = y[t==T_cond,1].item() + vfun(theta_0)
            y0_ = np.array([y[t==T_cond,0].item(),v_new])
            y_ = odeint(df_dt, y0_, t_, args = (l_s[i],))
        else:
            #if absolute_fun:
            #y0_ = y[t==10][0]
            #y_ = odeint(df_dt_fun,y0_,t_,args = (l_s[i],))
            #else:
            if strong_confounding:
                A_ = A * vfun(theta_0)
                A_ = A * theta_0
            else:
                A_ = A
            y0_ = np.concatenate((y[t==T_cond],np.array([0,1,0,A_])[None,:]),1)
            y_ = odeint(df_dt_complete,y0_[0],t_,args=(l_s[i],phi,delta))

        x = y[t<=T_cond,0]
        y_0 = y[t>T_cond,0]
        y_1 = y_[t_>T_cond,0]
        
        X.append(torch.Tensor(x))
        Y_0.append(torch.Tensor(y_0))
        Y_1.append(torch.Tensor(y_1))
        
    v_treatment = np.array(v_treatment)
    
    p = 1-sigmoid(gamma*(thetas_0-1))
    #p = sigmoid(gamma*(v_treatment-1))
    
    T = torch.zeros(N)
    if counterfactual:
        T[np.random.rand(N)>p] = 1
    else:
        T[np.random.rand(N)<p] = 1

    Y_0 = torch.stack(Y_0) + noise_std * torch.randn(N,len(t_y))
    Y_1 = torch.stack(Y_1) + noise_std * torch.randn(N,len(t_y))
    X = torch.stack(X) + noise_std * torch.randn(N,len(t_x))

    Y_fact = Y_0 * (1-T)[:,None] + Y_1 * T[:,None]
    Y_cf = Y_0 * (T)[:,None] + Y_1 * (1-T)[:,None]
    
    A_x = torch.zeros(X.shape)[...,None]
    A_x[T==1,-1] = 1 
    A_y = torch.zeros(Y_fact.shape)[...,None]

    if strong_confounding:
        T = T * thetas_0

    if static_x:
        # THIS IS HARDCODED SHIT THAT IS THERE JUST BECAUSE I WAS TOO LAZY TO HANDLE A CLEAN SOLUTION
        # returns only the non-treated occurence in the dataset
        treatment_mask = (T>=0)
        X = np.concatenate((thetas_0[treatment_mask,None],l_s[treatment_mask,None]),-1)
        X = X-X.mean(0)
        std_ = X.std(0)
        std_[std_==0] = 1
        X = X/(std_)
        return X , T[treatment_mask], Y_fact[treatment_mask,...,None], Y_cf[treatment_mask,...,None], p[treatment_mask], thetas_0[treatment_mask]
    
    X = X[...,None]
    Y_fact = Y_fact[...,None]
    Y_cf = Y_cf[...,None]
    M_before = torch.ones(X.shape)
    M_after = torch.ones(Y_fact.shape)
    
    times_X = torch.Tensor(t_x[None,:]).repeat(Y_fact.shape[0],1)
    times_Y = torch.Tensor(t_y[None,:]).repeat(Y_cf.shape[0],1)

    return X, T, Y_fact, Y_cf, p, thetas_0, times_X, times_Y, A_x, A_y, M_before, M_after

class PendulumDataset(Dataset):
    def __init__(self,N, gamma,noise_std, seed, continuous_treatment, fixed_length, static_x, strong_confounding, T_cond, T_horizon, counterfactual = False, dmm_order = False):

        X, T, Y_fact, Y_cf, p, X_0, times_X, times_Y, A_x, A_y, M_before, M_after = create_pendulum_data(N, gamma, noise_std, seed, continuous_treatment, fixed_length, static_x, strong_confounding, T_cond = T_cond, T_horizon = T_horizon, counterfactual = counterfactual)
       
        self.X = X.permute(0,2,1)
        self.T = T
        self.Y_fact = Y_fact.permute(0,2,1)
        self.Y_cf = Y_cf.permute(0,2,1)
        self.T_cf = (~T.bool()).float()
        self.p = p
        self.X_0 = torch.zeros(X.shape[0],3)
        self.A_x = A_x.permute(0,2,1)
        self.A_y = A_y.permute(0,2,1)
        self.M_before = M_before.permute(0,2,1)
        self.M_after = M_after.permute(0,2,1)
        
        self.conditional_dim = self.X.shape[1] + self.M_before.shape[1] + self.A_x.shape[1]
        self.output_dim = self.Y_fact.shape[1]
        self.conditional_len = self.X.shape[-1]
        self.baseline_size = self.X_0.shape[-1]
        self.treatment_dim = self.A_x.shape[1]
        self.ts_dim = self.X.shape[1]
        
        means_x = self.X.mean(dim=(0,2))[None,:,None]
        stds_x = self.X.std(dim=(0,2))[None,:,None]

        means_y = self.Y_fact.mean(dim=(0,2))[None,:,None]
        stds_y = self.Y_fact.std(dim=(0,2))[None,:,None]

        self.X = (self.X-means_x)/stds_x
        self.Y_fact = (self.Y_fact-means_y)/stds_y
        self.Y_cf = (self.Y_cf-means_y)/stds_y

        self.times_X = times_X
        self.times_Y = times_Y
        
        self.counterfactual = counterfactual
        self.dmm_order = dmm_order

        self.CE = torch.zeros(self.X.shape[0])
        self.Y_countdown = torch.zeros(self.X.shape[0])

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
            return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.X_0[idx], self.A_x[idx], self.A_y[idx], self.M_before[idx], self.M_after[idx], self.times_X[idx], self.times_Y[idx], self.Y_countdown[idx], self.CE[idx]

    def __len__(self):
        return self.X.shape[0]

class PendulumDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, num_workers = 4, continuous_treatment = False, fixed_length= False, static_x = False, strong_confounding = False, T_cond = 0, T_horizon = 0, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        if static_x:
            self.input_dim = 2
        else:
            self.input_dim = 1
        self.output_dim = 1

        self.N = N_ts
        self.gamma = gamma
        self.noise_std = noise_std

        self.continuous_treatment = continuous_treatment

        self.fixed_length = fixed_length

        self.static_x = static_x
        self.strong_confounding = strong_confounding

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        if kwargs["model_type"] == "DMM":
            self.dmm_order = True
        else:
            self.dmm_order = False

    def prepare_data(self):

        dataset = PendulumDataset(self.N, self.gamma, self.noise_std, self.seed, self.continuous_treatment, self.fixed_length, self.static_x, self.strong_confounding, T_cond = self.T_cond, T_horizon = self.T_horizon, dmm_order = self.dmm_order)       
        self.dataset_cf = PendulumDataset(self.N, self.gamma, self.noise_std, self.seed, self.continuous_treatment, self.fixed_length, self.static_x, self.strong_confounding, T_cond = self.T_cond, T_horizon = self.T_horizon, counterfactual  = True, dmm_order = self.dmm_order)  

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


class SyntheticMMDataModule(pl.LightningDataModule):
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
            data = TensorDataset(X, Y, T, Y_cf,p,  B[idx_sort], A_x,A_y, M_before, M_after, times_X, times_Y, Y_countdown, CE)
        
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
