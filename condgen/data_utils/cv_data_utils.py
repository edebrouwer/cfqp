
import pytorch_lightning as pl
from condgen.utils import DATA_DIR
#from causalode.datagen import cancer_simulation
import condgen.utils as utils
from condgen.utils import str2bool
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import argparse
import numpy as np
from scipy.integrate import odeint
import pandas as pd


def fluids_input(t):
    return 5*np.exp(-((t-5)/5)**2)

def v_fun(x):
    return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dx_dt(state, t, params):
    # Parameters:
    f_hr_max = params["f_hr_max"]
    f_hr_min = params["f_hr_min"]
    r_tpr_max = params["r_tpr_max"]
    r_tpr_min = params["r_tpr_min"]
    ca = params["ca"]
    cv = params["cv"]
    k_width = params["k_width"]
    p_aset = params["p_aset"]
    tau = params["tau"]
    t_treatment = params["t_treatment"]

    # Unknown parameters:
    group_factor = 1

    if (params["treatment"]) and (t>=t_treatment):
        initp_transform  = 0.5+(params["init_pressure"]-0.75)/0.1
        A_ = v_fun(initp_transform)
        if "treatment_value" in params:
            A_ *= params["treatment_value"]
            group_factor += 2*params["hidden_group"]
            if params["non_additive_noise"]:
                group_factor += np.random.randn()*params["noise_scale"]
        #A_ = 1
        i_ext = A_ * fluids_input(t-t_treatment)
    else:
        i_ext = 0
    
    
    r_tpr_mod = params["r_tpr_mod"]
    sv_mod = params["sv_mod"]

    # State variables
    p_a = 100. * state[0]
    p_v = 10. * state[1]
    s = state[2]
    sv = 100. * state[3]

    # Building f_hr and r_tpr:
    f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
    r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min - r_tpr_mod

    # Building dp_a/dt and dp_v/dt:
    dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
    dvv_dt = -1. * dva_dt + i_ext * group_factor
    dpa_dt = dva_dt / (ca * 100.)
    dpv_dt = dvv_dt / (cv * 10.)

    # Building dS/dt:
    ds_dt = (1. / tau) * (1. - 1. / (1 + np.exp(-1 * k_width * (p_a - p_aset))) - s)

    dsv_dt = i_ext * sv_mod

    # State derivative
    return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])

def init_random_state():
    max_ves = 64.0 - 10.0
    min_ves = 36.0 + 10.0

    max_ved = 167.0 - 10.0
    min_ved = 121.0 + 10.0

    max_sv = 1.0
    min_sv = 0.9

    max_pa = 85.0
    min_pa = 75.0

    max_pv = 7.0
    min_pv = 3.0

    max_s = 0.25
    min_s = 0.15

    init_ves = (np.random.rand() * (max_ves - min_ves) + min_ves) / 100.0
    # init_ves = 50.0 / 100.0

    init_ved = (np.random.rand() * (max_ved - min_ved) + min_ved) / 100.0
    # init_ved = 144.0 / 100.0

    init_sv = (np.random.rand() * (max_sv - min_sv) + min_sv)
    init_pa = (np.random.rand() * (max_pa - min_pa) + min_pa) / 100.0
    init_pv = (np.random.rand() * (max_pv - min_pv) + min_pv) / 10.0
    init_s = (np.random.rand() * (max_s - min_s) + min_s)

    init_state = np.array([init_pa, init_pv, init_s, init_sv])
    return init_state


def create_grouped_cv_data(noise_std, t_span = 30, t_treatment = 15, seed = 421, normalize = True, output_dims = [1], input_dims = [0,1], full_output = False , delta_t = 1, Z = None, T = None, non_additive_noise = False):

    np.random.seed(seed)

    X = []
    Y_0 = []
    init_state_list = []
    
    params = {"r_tpr_mod": 0.,
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "sv_mod": 0.001,
            "ca": 4.0,
            "cv": 111.0,

            # dS/dt parameters
            "k_width": 0.1838,
            "p_aset": 70,
            "tau": 20,
            "p_0lv": 2.03,
            "r_valve": 0.0025,
            "k_elv": 0.066,
            "v_ed0": 7.14,
            "T_sys": 4. / 15.,
            "cprsw_max": 103.8,
            "cprsw_min": 25.9,
            "t_treatment" : t_treatment,
            "non_additive_noise":  non_additive_noise,
            "noise_scale":noise_std
            }
    
    params_treatment = params.copy()
    params_treatment["treatment"]=True
    params_notreatment = params.copy()
    params_notreatment["treatment"]=False
    
    t = np.arange(t_span+1).astype(float)
    if delta_t == 0.5:
        t = np.arange(2*t_span+1)/2
    
    init_state = init_random_state()
    params_treatment["init_pressure"] = init_state[0]
    params_treatment["treatment_value"] = T
    params_treatment["hidden_group"] = Z
    params_treatment["cv"] = np.random.rand() * 100 + 10
    y0 = odeint(dx_dt,init_state,t,args=tuple([params_treatment]))
        
    X.append(torch.Tensor(init_state))
    Y_0.append(torch.Tensor(y0[1:]))
    init_state_list.append(torch.Tensor(init_state))
    
    init_state = torch.stack(init_state_list)

    Y_0 = torch.stack(Y_0)
    X = torch.stack(X)
    if not non_additive_noise:
        Y_0 += noise_std * torch.randn(Y_0.shape)
        X += noise_std * torch.randn(X.shape)

    Y_fact = Y_0

    if normalize:
        mu = Y_fact.mean([0,1])
        std = Y_fact.std([0,1])

        Y_fact = (Y_fact - mu)/std
        
        mu_X = X.mean([0,1])
        std_X = X.std([0,1])
        X = (X-mu_X)/std_X
   
    pre_treat_mask = (t<=t_treatment)[1:]
    post_treat_mask = (t>t_treatment)[1:]
    
    X_static = X
    X = Y_fact[:,pre_treat_mask][:,:,input_dims]

    Y_fact_full = Y_fact[:,post_treat_mask][:,:,input_dims]
    if not full_output:
        Y_fact = Y_fact[:,post_treat_mask][:,:,output_dims] 
    else:
        Y_fact = Y_fact[:,post_treat_mask][:,:,input_dims] 


    A_x = torch.zeros(X.shape[:-1])[...,None]
    A_x[T==1,-1] = 1
    A_y = torch.zeros(Y_fact.shape[:-1])[...,None]

    M_x = torch.ones(X.shape)
    M_y = torch.ones(Y_fact.shape)
    M_y_full = torch.ones(Y_fact_full.shape)

    return X, Y_fact, t[1:][pre_treat_mask],t[1:][post_treat_mask]

def create_cv_data(N,gamma,noise_std, t_span = 30, t_treatment = 15, seed = 421, normalize = True, output_dims = [1], input_dims = [0,1], full_output = False , delta_t = 1, counterfactual = False, non_additive_noise = False):

    np.random.seed(seed)

    X = []
    Y_0 = []
    Y_1 = []
    init_state_list = []
    
    params = {"r_tpr_mod": 0.,
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "sv_mod": 0.001,
            "ca": 4.0,
            "cv": 111.0,

            # dS/dt parameters
            "k_width": 0.1838,
            "p_aset": 70,
            "tau": 20,
            "p_0lv": 2.03,
            "r_valve": 0.0025,
            "k_elv": 0.066,
            "v_ed0": 7.14,
            "T_sys": 4. / 15.,
            "cprsw_max": 103.8,
            "cprsw_min": 25.9,
            "t_treatment" : t_treatment,
            "non_additive_noise" : non_additive_noise
            }
    
    params_treatment = params.copy()
    params_treatment["treatment"]=True
    params_notreatment = params.copy()
    params_notreatment["treatment"]=False
    
    t = np.arange(t_span+1).astype(float)
    if delta_t == 0.5:
        t = np.arange(2*t_span+1)/2
    
    for i in range(N):
        init_state = init_random_state()
        params_treatment["init_pressure"] = init_state[0]
        params_treatment["cv"] = np.random.rand() * 100 + 10
        y1 = odeint(dx_dt,init_state,t,args=tuple([params_treatment]))
        
        params_notreatment["init_pressure"] = init_state[0]
        params_notreatment["cv"] = params_treatment["cv"]
        y0 = odeint(dx_dt,init_state,t,args=tuple([params_notreatment]))
        
        X.append(torch.Tensor(init_state))
        Y_0.append(torch.Tensor(y0[1:]))
        Y_1.append(torch.Tensor(y1[1:]))
        init_state_list.append(torch.Tensor(init_state))
    
    init_state = torch.stack(init_state_list)
    p = torch.sigmoid(gamma*((init_state[:,0]-0.75)/0.1-0.5))
    T = torch.zeros(N)
    if counterfactual:
        T[torch.rand(N)>p] = 1
    else:
        T[torch.rand(N)<p] = 1

    Y_0 = torch.stack(Y_0)
    Y_1 = torch.stack(Y_1)
    X = torch.stack(X)
    if not non_additive_noise:
        Y_0 += noise_std * torch.randn(Y_0.shape)
        Y_1 += noise_std * torch.randn(Y_1.shape)
        X += noise_std * torch.randn(X.shape)

    Y_fact = Y_0 * (1-T)[:,None,None] + Y_1 * T[:,None,None]
    Y_cf = Y_0 * (T)[:,None,None] + Y_1 * (1-T)[:,None,None]

    if normalize:
        mu = Y_fact.mean([0,1])
        std = Y_fact.std([0,1])

        Y_fact = (Y_fact - mu)/std
        Y_cf = (Y_cf - mu)/std
        
        mu_X = X.mean([0,1])
        std_X = X.std([0,1])
        X = (X-mu_X)/std_X
   
    pre_treat_mask = (t<=t_treatment)[1:]
    post_treat_mask = (t>t_treatment)[1:]
    
    X_static = X
    X = Y_fact[:,pre_treat_mask][:,:,input_dims]
    X_ = Y_cf[:,pre_treat_mask][:,:,input_dims]

    Y_fact_full = Y_fact[:,post_treat_mask][:,:,input_dims]
    if not full_output:
        Y_fact = Y_fact[:,post_treat_mask][:,:,output_dims] 
        Y_cf = Y_cf[:,post_treat_mask][:,:,output_dims]
    else:
        Y_fact = Y_fact[:,post_treat_mask][:,:,input_dims] 
        Y_cf = Y_cf[:,post_treat_mask][:,:,input_dims]


    A_x = torch.zeros(X.shape[:-1])[...,None]
    A_x[T==1,-1] = 1
    A_y = torch.zeros(Y_fact.shape[:-1])[...,None]

    M_x = torch.ones(X.shape)
    M_y = torch.ones(Y_fact.shape)
    M_y_full = torch.ones(Y_fact_full.shape)

    return X, T, Y_fact, Y_cf, p, X_static, t[1:][pre_treat_mask],t[1:][post_treat_mask], A_x, A_y, M_x, M_y, output_dims, Y_fact_full, M_y_full



class CVDataset(Dataset):
    def __init__(self,N, gamma,noise_std, T_cond, T_horizon, seed, full_output, delta_t, counterfactual, dmm_order  = False ):

        X, T, Y_fact, Y_cf, p, X_0, times_X, times_Y, A_x, A_y, M_x, M_y, output_dims, Y_fact_full, M_y_full = create_cv_data(N = N, gamma = gamma, noise_std = noise_std, t_span = T_cond+T_horizon, t_treatment = T_cond, seed = seed, full_output = full_output, delta_t = delta_t, counterfactual = counterfactual)

        self.X = X.permute(0,2,1)
        self.T = T
        self.Y_fact = Y_fact.permute(0,2,1)
        self.Y_fact_full = Y_fact_full.permute(0,2,1)
        self.Y_cf = Y_cf.permute(0,2,1)
        self.T_cf = (~T.bool()).float()
        self.p = p
        self.X_0 = X_0
        self.A_x = A_x.permute(0,2,1)
        self.A_y = A_y.permute(0,2,1)
        self.M_x = M_x.permute(0,2,1)
        self.M_y = M_y.permute(0,2,1)
        self.M_y_full = M_y_full.permute(0,2,1)

        self.conditional_dim = self.X.shape[1] + self.A_x.shape[1] + self.M_x.shape[1]
        self.conditional_len = self.X.shape[-1]
        if full_output:
            self.output_dim = 2 #HARD CODED THE DIMENSION
        else:
            self.output_dim = self.Y_fact.shape[1]
        
        self.baseline_size = self.X_0.shape[-1]
        self.treatment_dim = self.A_x.shape[1]
        self.ts_dim = self.X.shape[1]

        self.times_X = torch.Tensor(times_X[None,:]).repeat(Y_fact.shape[0],1)
        self.times_Y = torch.Tensor(times_Y[None,:]).repeat(Y_cf.shape[0],1)

        self.output_dims = output_dims

        self.dmm_order = dmm_order
        if self.dmm_order:
            self.B = self.X_0
            self.X = torch.cat((self.X,self.Y_fact_full),-1).permute(0,2,1)
            self.A = torch.cat((self.A_x,self.A_y),-1).permute(0,2,1)
            self.M = torch.cat((self.M_x,self.M_y_full),-1).permute(0,2,1)
            self.CE = torch.zeros(self.X.shape[0])
            self.events = torch.zeros(self.X.shape[0],self.X.shape[1])

    def __getitem__(self,idx):
        if self.dmm_order:
            return self.B[idx], self.X[idx], self.A[idx], self.M[idx], self.events[idx], self.CE[idx]
        else:
            return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.X_0[idx], self.A_x[idx], self.A_y[idx], self.M_x[idx], self.M_y[idx], self.times_X[idx], self.times_Y[idx]
    def __len__(self):
        return self.X.shape[0]

class CVDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, T_cond, T_horizon, num_workers = 4, non_additive_noise =False, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 2
        self.output_dim = 1 # number of dimensions to reconstruct in the time series

        self.N = N_ts
        self.gamma = gamma
        self.noise_std = noise_std
        self.T_cond = T_cond
        self.T_horizon = T_horizon
        self.non_additive_noise = non_additive_noise

        if kwargs["model_type"] == "Transformer":
            self.full_output = True
        else:
            self.full_output = False
        
        if kwargs["model_type"] == "ScoreMatching":
            self.delta_t = 0.5
        else:
            self.delta_t = 1

        if kwargs["model_type"] == "DMM":
            self.dmm_order = True
        else:
            self.dmm_order = False

    def prepare_data(self):

        dataset = CVDataset(N = self.N, gamma = self.gamma,noise_std =  self.noise_std, seed = self.seed, T_cond = self.T_cond, T_horizon = self.T_horizon, full_output = self.full_output, delta_t = self.delta_t, counterfactual = False, dmm_order = self.dmm_order)       
        self.cf_dataset = CVDataset(N = self.N, gamma = self.gamma,noise_std =  self.noise_std, seed = self.seed, T_cond = self.T_cond, T_horizon = self.T_horizon, full_output = self.full_output, delta_t = self.delta_t, counterfactual = True, dmm_order = self.dmm_order)       
        
        train_idx = np.arange(len(dataset))[:int(0.5*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]


        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)

        self.conditional_dim = dataset.conditional_dim
        self.conditional_len = dataset.conditional_len
        self.output_dim = dataset.output_dim
        self.treatment_dim = dataset.treatment_dim
        self.baseline_size = dataset.baseline_size
        self.output_dims = dataset.output_dims
        self.ts_dim = dataset.ts_dim
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

    def cf_test_dataloader(self):
        return DataLoader(
            self.cf_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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
        parser.add_argument('--non_additive_noise', type=str2bool, default=False, help = "If true, adds non-additive noise to the data")
        parser.add_argument('--T_cond', type=int, default=15, help = "the length of the data to consider for conditioning")
        parser.add_argument('--T_horizon', type=int, default=11, help = "the length of the predictions")
        return parser

   
if __name__=="__main__":
    
    datam = CVDataModule(batch_size=32, N_ts = 1000, gamma = 0, seed = 4221, noise_std = 0., num_workers = 0 )
    datam.prepare_data()
    
    #dataset = FluidsDataset(path="/home/edward/Data/pre_processed_mimic/models/")
    #datam = FluidDataModule(path = "/home/edward/Data/pre_processed_mimic/models/")

    #datam.prepare_data()

    for i,b in enumerate(datam.train_dataloader()):
        print(b)
        import ipdb; ipdb.set_trace()
        #break

    #dataset = CancerDataModule(batch_size = 32, seed = 23, chemo_coeff = 2, radio_coeff = 2, window_size = 15, num_time_steps = 20, t_limit = 10, num_workers = 4, )
    #dataset.prepare_data()
