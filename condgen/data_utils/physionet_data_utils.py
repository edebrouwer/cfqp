import pytorch_lightning as pl
import sys
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

class PhysioDataset(Dataset):
    def __init__(self, T_cond, T_horizon, seed, full_output, dmm_order  = False, **kwargs ):
        
        self.data_pathA = f"{DATA_DIR}/physionet.org/files/challenge-2019/1.0.0/training/training_setA"
        self.data_pathB = f"{DATA_DIR}/physionet.org/files/challenge-2019/1.0.0/training/training_setB"
        
        dict_A = {i: os.path.join(self.data_pathA,f) for i,f in enumerate(os.listdir(self.data_pathA)) if ".psv" in f}
        idx_max_A = np.array([*dict_A]).max()
        dict_B = {i+idx_max_A: os.path.join(self.data_pathA,f) for i,f in enumerate(os.listdir(self.data_pathA)) if ".psv" in f}
        dict_A.update(dict_B)
        self.idx_dict = dict_A

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        self.longitudinal_vars = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']

        self.static_vars = [ 'Age', 'Gender','HospAdmTime']
        
        self.long_means = 0
        self.long_stds = 1
        self.static_means = 0
        self.static_stds = 1

        X, Y, T, Y, p, B, A_x, A_y, M_x, M_y, times_X, times_Y = self.__getitem__(0)

        self.conditional_dim = X.shape[0] + A_x.shape[0] + M_x.shape[0]
        self.conditional_len = T_cond
        self.output_dim = Y.shape[0]
        self.treatment_dim = A_x.shape[0]
        self.baseline_size = B.shape[0]
        self.output_dims = np.arange(self.output_dim)
        self.ts_dim = X.shape[0]
        
        self.filter_patients() # remove patients that have less than T_cond measurements 

    def filter_patients(self):
        remove_idx = []
        for idx in self.idx_dict.keys():
            df = pd.read_csv(self.idx_dict[idx],sep = "|")
            if len(df) <= (self.T_cond + 1):
                remove_idx.append(idx)
        for idx in remove_idx:
            del self.idx_dict[idx]
        print(f"Removed {len(remove_idx)} patients with too short time series")

    def get_idx(self):
        return np.array([*self.idx_dict])

    def init_normalizer(self,train_idx):
        df_list = []
        for idx in train_idx:
            df = pd.read_csv(self.idx_dict[idx],sep = "|")
            df_list.append(df)
        
        df_train = pd.concat(df_list)
        self.long_means = df_train[self.longitudinal_vars].mean(0).values
        self.long_stds = df_train[self.longitudinal_vars].std(0).values

        self.static_means = df_train[self.static_vars].mean(0).values
        self.static_stds = df_train[self.static_vars].std(0).values

  
    def process_df(self, df):
        if self.longitudinal_vars[0] not in df.columns:
            import ipdb; ipdb.set_trace()
        df_X = df[self.longitudinal_vars]
        df_B = df[self.static_vars]

        df_X_ = (df_X.values - self.long_means)/self.long_stds

        X = df_X_[:self.T_cond].T
        Y = df_X_[self.T_cond:self.T_cond+self.T_horizon].T
       
        times_X = df["ICULOS"].values[:self.T_cond]
        times_Y = df["ICULOS"].values[self.T_cond:self.T_cond+self.T_horizon]
        
        B = (df_B.iloc[0].values - self.static_means) / self.static_means
        A_x = np.zeros((1,self.T_cond))
        A_y = np.zeros((1,self.T_horizon))
        
        M_x = (~np.isnan(X)).astype(float)
        M_y = (~np.isnan(Y)).astype(float)

        X[np.isnan(X)] = 0
        Y[np.isnan(Y)] = 0

        if Y.shape[1]<self.T_horizon:
            M_y = np.concatenate((M_y,np.zeros((Y.shape[0],self.T_horizon-Y.shape[1]))),axis=1)
            times_Y = np.concatenate((times_Y,times_Y[-1]+np.arange(self.T_horizon-Y.shape[1])))
            Y = np.concatenate((Y,np.zeros((Y.shape[0],self.T_horizon-Y.shape[1]))),axis = 1)

        
        assert(Y.shape[1]==self.T_horizon)
        p = np.zeros(1)
        T = np.zeros(1)

        return torch.Tensor(X), torch.Tensor(Y), T, p, None, torch.Tensor(B), torch.Tensor(A_x), torch.Tensor(A_y), torch.Tensor(M_x), torch.Tensor(M_y), torch.Tensor(times_X), torch.Tensor(times_Y)


    def __getitem__(self,idx):
        df = pd.read_csv(self.idx_dict[idx],sep = "|")
        X, Y, T, p, _, B, A_x, A_y, M_x, M_y, times_X, times_Y = self.process_df(df)
        return X, Y, T, Y, p, B, A_x, A_y, M_x, M_y, times_X, times_Y
    
    def __len__(self):
        return len(self.idx_dict)


#def collate_fn(batch):
#    import ipdb; ipdb.set_trace()

class PhysioDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, T_cond, T_horizon, num_workers = 0, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.output_dim = 1 # number of dimensions to reconstruct in the time series

        self.T_cond = T_cond
        self.T_horizon = T_horizon
        
    def prepare_data(self):

        dataset = PhysioDataset(T_cond = self.T_cond, T_horizon = self.T_horizon, seed = self.seed, full_output = True)

        
        idx = dataset.get_idx()
        rng = np.random.default_rng(seed = self.seed)
        rng.shuffle(idx)

        train_idx = idx[:int(0.5*len(dataset))]
        val_idx = idx[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        dataset.init_normalizer(train_idx) # init the normalizer to use only the training data

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
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--T_cond', type=int, default=15, help = "the length of the data to consider for conditioning")
        parser.add_argument('--T_horizon', type=int, default=11, help = "the length of the predictions")
        return parser

if __name__ == "__main__":
    #ds = PhysioDataset(T_cond = 5, T_horizon = 10, seed = 421, full_output = True)
    #ds[3]

    dl = PhysioDataModule( batch_size = 32, T_cond = 5, T_horizon = 10, seed = 421)
    dl.prepare_data()
    for i,b in enumerate(dl.val_dataloader()):
        import ipdb; ipdb.set_trace()
