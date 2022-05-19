import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, VisionDataset
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image

from condgen.utils import str2bool
from condgen.data_utils.cv_data_utils import create_grouped_cv_data

def generate_ts_old(base, t_max,t_treat,deltat_plateau,T, phase ):

    t = np.arange(0,t_max)
    mask_x = t<=t_treat
    mask_y = t>t_treat

    alpha = 0.01

    x = base * np.ones(mask_x.sum()) + alpha*np.sin(t[mask_x])

    diff = 0.5
    plateau = deltat_plateau
    t_plateau = t[mask_y][:plateau]
    t_end = t[mask_y][plateau:]

    treatment_effect = T*diff
    plateau_y = ((np.arange(plateau)/plateau) * treatment_effect) + x[-1]  
    end_y = np.ones(mask_y.sum()-plateau) * (treatment_effect + x[-1]) #+ alpha*np.sin(t_end+phase)
    alpha = 0.1
    y = np.concatenate((plateau_y,end_y)) + alpha*(np.arange(mask_y.sum())/mask_y.sum())*np.sin(t[mask_y]+phase)
    
    return x, y ,t[mask_x], t[mask_y]

def generate_ts( t_max,t_treat,deltat_plateau,T, phase, group = 0, noise_scale = 0.05, non_additive_noise = False):

    t = np.arange(0,t_max)
    mask_x = t<=t_treat
    mask_y = t>t_treat
    
    x = np.stack((np.sin(0.5*t + phase),np.sin(0.5*t + phase *2)))
    diff = 1
    
    plateau = np.arange(deltat_plateau) * T * diff / deltat_plateau
    end_y = np.ones(mask_y.sum()-deltat_plateau) * T * diff
    diff = np.concatenate((plateau, end_y))
    
    if group==0:
        x[0,mask_y] += diff
    elif group==1:
        x[1,mask_y] += diff
    elif group==2:
        x[0,mask_y] += diff
        x[1,mask_y] += diff
    else:
        raise("Invalid group")

    if not non_additive_noise:
        x += noise_scale * np.random.randn(*x.shape)
        
    y = x[:,mask_y]
    x = x[:,mask_x]
    
    return x, y ,t[mask_x], t[mask_y]

def generate_cf_ts( t_max,t_treat,deltat_plateau,T_list, phase, group = 0, noise_scale = 0.05 ):
    return [generate_ts(t_max,t_treat,deltat_plateau, T, phase, group  = group, noise_scale = noise_scale) for T in T_list]
 
class SimpleTraj(Dataset):
    def __init__(self,N, random_seed = 421, non_additive_noise = False, noise_scale = 0.05):
        super().__init__()
        
        np.random.seed(random_seed)
        self.T = np.random.rand(N) * 0.8 + 0.2
        self.categorical_noise = torch.LongTensor(np.random.randint(3,size = (N)))[:,None]
        self.group = np.array([0,1,2])[self.categorical_noise]

        x_ = []
        y_ = []

        self.t_max = 40
        self.t_treat = 19
        self.deltat_plateau = 3
        self.noise_scale = noise_scale
        self.non_additive_noise = non_additive_noise
        for i in range(N):
            phase = np.random.randn()
            phase_vec = np.ones(self.t_max)
            if self.non_additive_noise:
                phase = np.random.rand() * 3 + 0.5
                phase_vec = np.ones(self.t_max)
                phase_vec += np.random.randn(self.t_max)*self.noise_scale
            
            x,y, t_x, t_y =  generate_ts(t_max =self.t_max,t_treat =self.t_treat,deltat_plateau = self.deltat_plateau,T = self.T[i], group = self.group[i], phase = phase_vec, noise_scale = self.noise_scale, non_additive_noise = self.non_additive_noise )
            x_.append(x)
            y_.append(y)

        self.X =  torch.Tensor(np.stack(x_))
        self.Y = torch.Tensor(np.stack(y_))
       
        self.mean_ = self.X.mean()
        self.std_  = self.X.std()
        self.X = (self.X - self.mean_) / self.std_
        self.Y = (self.Y - self.mean_) / self.std_
        
        self.N = N

    #def generate_cf_ts(self,T_list, phase, group = 0 ):
    #    out_list = []
    #    for T in T_list:
    #        x,y,t_x,t_y = generate_ts(t_max = self.t_max,t_treat = self.t_treat,deltat_plateau = self.deltat_plateau, T = T, phase = phase, group  = group, noise_scale = self.noise_scale)
    #        x = (x - self.mean_.numpy()) / self.std_.numpy()
    #        y = (y - self.mean_.numpy()) / self.std_.numpy()
    #        out_list.append((x,y,t_x,t_y))
    #    return out_list
    
    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx], torch.zeros(1), self.categorical_noise[idx], self.T[idx]

class SimpleTrajCF(Dataset):
    def __init__(self,N, means = None, stds = None, random_seed = 521, noise_scale = 0.05, non_additive_noise = False):
        super().__init__()
        
        np.random.seed(random_seed)

        self.T_o = np.random.rand(N) * 0.8 + 0.2
        self.T_new = np.random.rand(N) * 0.8 + 0.2

        self.categorical_noise = torch.LongTensor(np.random.randint(3,size = (N)))[:,None]
        self.group = np.array([0,1,2])[self.categorical_noise]

        x_ = []
        yo_ = []
        ynew_ = []
        
        self.t_max = 40
        self.t_treat = 19
        self.deltat_plateau = 3
        self.noise_scale = noise_scale
        self.non_additive_noise = non_additive_noise

        for n in range(N):

            out_list = self.generate_cf_ts(T_list = [self.T_o[n],self.T_new[n]],phase = np.random.randn(), group = self.group[n])

            (x_o,y_o,t_x_o, t_y_o) = out_list[0]
            (x_new,y_new,t_x_new, t_y_new) = out_list[1]

            x_.append(x_o)
            yo_.append(y_o)
            ynew_.append(y_new)

        self.X =  torch.Tensor(np.stack(x_))
        self.Y_o = torch.Tensor(np.stack(yo_))
        self.Y_new = torch.Tensor(np.stack(ynew_))

        if means is None:
            self.mean_ = self.X.mean()
            self.std_  = self.X.std()
        else:
            self.mean_ = means
            self.std_ = stds

        self.X = (self.X - self.mean_) / self.std_
        self.Y_o = (self.Y_o - self.mean_) / self.std_
        self.Y_new = (self.Y_new - self.mean_) / self.std_
        
        self.N = N

    def generate_cf_ts(self,T_list, phase, group = 0 ):
        out_list = []
        phase_vec = np.ones(self.t_max)
        if self.non_additive_noise:
            phase = np.random.rand() * 3 + 0.5
            phase_vec = np.ones(self.t_max)
            phase_vec += np.random.randn(self.t_max)*self.noise_scale
        for T in T_list:
            x,y,t_x,t_y = generate_ts(t_max = self.t_max,t_treat = self.t_treat,deltat_plateau = self.deltat_plateau, T = T, phase = phase_vec, group  = group, noise_scale = self.noise_scale, non_additive_noise = self.non_additive_noise)
            out_list.append((x,y,t_x,t_y))
        return out_list
    
    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        return self.X[idx], self.Y_o[idx], self.categorical_noise[idx], self.T_o[idx], self.T_new[idx], self.Y_new[idx]


class CVTraj(Dataset):
    def __init__(self,N, random_seed = 421, noise_scale = 0., non_additive_noise = False):
        super().__init__()
        
        np.random.seed(random_seed)
        self.T = np.random.rand(N) * 0.6 + 0.4
        self.group = torch.multinomial(input = torch.ones(N,2)*0.5, num_samples = 1)
        self.noise = (100*torch.rand(N)).long()

        x_ = []
        y_ = []

        self.t_max = 40
        self.t_treat = 20
        self.noise_scale = noise_scale
        self.non_additive_noise = non_additive_noise

        for i in range(N):
            x,y,t_x, t_y = create_grouped_cv_data(noise_std = self.noise_scale, t_span = self.t_max, t_treatment = self.t_treat, normalize = False, output_dims = [0,1], input_dims = [0,1], full_output = False, delta_t = 1, Z = self.group[i].item(), T = self.T[i], seed = self.noise[i])
            x_.append(x)
            y_.append(y)

        self.X =  torch.Tensor(np.stack(x_))[:,0,...].permute(0,2,1)
        self.Y = torch.Tensor(np.stack(y_))[:,0,...].permute(0,2,1)
       
        self.mean_ = self.X.mean()
        self.std_  = self.X.std()
        self.X = (self.X - self.mean_) / self.std_
        self.Y = (self.Y - self.mean_) / self.std_
        
        self.N = N
        
    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx], torch.zeros(1), self.group[idx], self.T[idx]

class CVTrajCF(Dataset):
    def __init__(self,N, means = None, stds = None, random_seed = 521, noise_scale = 0., non_additive_noise = False, ite_mode = False, treatment2 = None, treatment3 = None):
        super().__init__()
        
        np.random.seed(random_seed)

        self.T_o = np.random.rand(N) * 0.6 + 0.4
        self.T_new = np.random.rand(N) * 0.6 + 0.4
        self.T_new2 = np.random.rand(N) * 0.6 + 0.4

        if ite_mode:
            self.T_new = np.ones(N) * treatment2
            self.T_new2 = np.ones(N) * treatment3
        self.ite_mode = ite_mode

        self.group = torch.multinomial(input = torch.ones(N,2)*0.5, num_samples = 1)
        self.noise = (100*torch.rand(N)).long()


        x_ = []
        yo_ = []
        ynew_ = []
        ynew2_ = []
        
        self.t_max = 40
        self.t_treat = 20
        self.noise_scale = noise_scale
        self.non_additive_noise = non_additive_noise
        
        for n in range(N):
            
            out_list = self.generate_cf_ts(T_list = [self.T_o[n],self.T_new[n],self.T_new2[n]], noise = self.noise[n], group = self.group[n].item())

            (x_o,y_o,t_x_o, t_y_o) = out_list[0]
            (x_new,y_new,t_x_new, t_y_new) = out_list[1]
            (x_new2,y_new2,t_x_new2, t_y_new2) = out_list[2]

            x_.append(x_o)
            yo_.append(y_o)
            ynew_.append(y_new)
            ynew2_.append(y_new2)

        self.X =  torch.Tensor(np.stack(x_))[:,0,...].permute(0,2,1)
        self.Y_o = torch.Tensor(np.stack(yo_))[:,0,...].permute(0,2,1)
        self.Y_new = torch.Tensor(np.stack(ynew_))[:,0,...].permute(0,2,1)
        self.Y_new2 = torch.Tensor(np.stack(ynew2_))[:,0,...].permute(0,2,1)

        if means is None:
            self.mean_ = self.X.mean()
            self.std_  = self.X.std()
        else:
            self.mean_ = means
            self.std_ = stds

        self.X = (self.X - self.mean_) / self.std_
        self.Y_o = (self.Y_o - self.mean_) / self.std_
        self.Y_new = (self.Y_new - self.mean_) / self.std_
        self.Y_new2 = (self.Y_new2 - self.mean_) / self.std_
        
        self.N = N

    def generate_cf_ts(self,T_list, group , noise):
        out_list = []
        for T in T_list:
            x,y,t_x, t_y = create_grouped_cv_data(noise_std = self.noise_scale, t_span = self.t_max, t_treatment = self.t_treat, normalize = False, output_dims = [0,1], input_dims = [0,1], full_output = False, delta_t = 1, Z = group, T = T, seed = noise)
            out_list.append((x,y,t_x,t_y))
        return out_list
    
    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        if self.ite_mode:
            return self.X[idx], self.Y_o[idx], self.group[idx], self.T_o[idx], self.T_new[idx], self.Y_new[idx], self.T_new2[idx], self.Y_new2[idx]
        else:
            return self.X[idx], self.Y_o[idx], self.group[idx], self.T_o[idx], self.T_new[idx], self.Y_new[idx]


class SimpleTrajDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, random_seed, num_workers = 4, num_classes = 3, data_type = "SimpleTraj", noise_std = 0., non_additive_noise = False, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.train_shuffle = True
        
        if data_type == "CV":
            num_classes = 2
        self.num_classes = num_classes #number of colors
        
        self.data_type = data_type
        self.noise_std = noise_std
        self.non_additive_noise = non_additive_noise

    def prepare_data(self, ite_mode = False, treatment2 = None, treatment3 = None):
        
        if self.data_type == "SimpleTraj":
            dataset = SimpleTraj(3000, random_seed = self.random_seed, non_additive_noise = self.non_additive_noise, noise_scale = self.noise_std)
            self.cf_dataset = SimpleTrajCF(1000, random_seed = self.random_seed + 100, non_additive_noise = self.non_additive_noise, noise_scale = self.noise_std)
        elif self.data_type == "CV":
            dataset = CVTraj(5000, random_seed = self.random_seed, noise_scale = self.noise_std, non_additive_noise = self.non_additive_noise)
            self.cf_dataset = CVTrajCF(1000, random_seed = self.random_seed + 100, means = dataset.mean_, stds = dataset.std_, noise_scale = self.noise_std, non_additive_noise = self.non_additive_noise, ite_mode = ite_mode, treatment2 =treatment2, treatment3 = treatment3)

        train_idx = np.arange(len(dataset))[:int(0.7*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.7*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        self.train_batch_size = self.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size

        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)

        
        self.input_dim = dataset.X.shape[1]
        self.output_dim = dataset.Y.shape[1]
        self.conditional_len = dataset.X.shape[-1]

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

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

    def test_cf_dataloader(self):
        return DataLoader(
            self.cf_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )
        
    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--max_angle', type=float, default=30)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_classes', type=int, default=3)
        parser.add_argument('--noise_std', type=float, default=0.)
        parser.add_argument('--non_additive_noise', type=str2bool, default=False, help = "If true, adds non-additive noise to the data")
        parser.add_argument('--colored', type=str2bool, default=False, help = "If true, uses the colored - rotated MNIST version")
        return parser

if __name__ == "__main__":
    dataset = ColoredMNIST(root='./playground')
    img, y, color = dataset[0]
