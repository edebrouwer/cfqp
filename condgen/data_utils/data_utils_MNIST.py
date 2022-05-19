import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, VisionDataset
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image
import os
from condgen.utils import str2bool
from condgen.utils import DATA_DIR

def color_grayscale_arr(arr, color):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  assert(color.max()<6) #Should create new sampling points first
  
  props = torch.Tensor([[0.,0.,1.],[0.5,0.,0.5],[1.,0.,0.],[0.,0.5,0.5],[0.,1.,0.],[0.5,0.5,0.]])
  #props = props[:num_classes]
  #import ipdb; ipdb.set_trace()
  return (props[color][None,:] * arr).type(dtype)



def multinomial_probability(targets, confounding_strength, num_classes):
    nunique_targets = len(torch.unique(targets))
    average_prob = 1/num_classes
    outstanding_prob = (1-average_prob) * confounding_strength + average_prob
    probs_vec = torch.ones(len(targets),num_classes) * (1-outstanding_prob)/(num_classes-1)
    for idx in range(len(targets)):
        probs_vec[idx,targets[idx]%num_classes] = outstanding_prob
    return probs_vec

class ColoredMNIST(MNIST):
    def __init__(self,root, train = True, transform = None, target_transform = None, download = True, max_angle = 30, num_classes = 6,seed = 421, non_additive_noise = False, noise_std = 0., w_confounding = 0.):
        super(ColoredMNIST,self).__init__(root = root, train = train, transform = None, target_transform = None, download = download)
        
        torch.manual_seed(seed)

        noise_probs = multinomial_probability(self.targets,confounding_strength = w_confounding, num_classes = num_classes)
        self.categorical_noise = torch.multinomial(input = noise_probs, num_samples = 1)
        #self.categorical_noise = torch.randint(low = 0, high = num_classes, size = (self.data.shape[0],1))
        self.treatment = torch.rand(self.data.shape[0])*max_angle + torch.sigmoid((self.data.float().mean((1,2))-33)/11) * 5
        self.to_tensor = transforms.ToTensor()
        
        self.noise_std = noise_std
        self.non_additive_noise = non_additive_noise


    def __getitem__(self,idx):
        img, target = self.data[idx], int(self.targets[idx])
        
        img_og = Image.fromarray(img.numpy())

        img = color_grayscale_arr(img, color = self.categorical_noise[idx])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())#, mode='L')

        if self.non_additive_noise:
            blurrer = transforms.GaussianBlur(kernel_size = (5,5), sigma= (self.noise_std))
            img = blurrer(img)
        
        img = transforms.functional.rotate(img, angle = self.treatment[idx].item())
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img_og = self.to_tensor(img_og)
        img = self.to_tensor(img)

        return img_og, img, target, self.categorical_noise[idx], self.treatment[idx]


class ColoredMNISTCF(MNIST):
    def __init__(self,root, train = True, transform = None, target_transform = None, download = True, max_angle = 30, num_classes = 6,seed = 421, non_additive_noise = False, noise_std = 0., w_confounding = 0., ite_mode = False, treatment2 = None, treatment3 = None):
        super(ColoredMNISTCF,self).__init__(root = root, train = train, transform = None, target_transform = None, download = download)
       
        self.data = self.data[:1000]
        torch.manual_seed(seed)
        noise_probs = multinomial_probability(self.targets,confounding_strength = w_confounding, num_classes = num_classes)
        self.categorical_noise = torch.multinomial(input = noise_probs, num_samples = 1)
        #self.categorical_noise = torch.randint(low = 0, high = num_classes, size = (self.data.shape[0],1))
        self.treatment = torch.rand(self.data.shape[0]) * max_angle + torch.sigmoid((self.data.float().mean((1,2))-33)/11) * 5
        if ite_mode:
            assert(treatment2 is not None)
            self.treatment2 = torch.ones(self.data.shape[0]) * treatment2
            self.treatment3 = torch.ones(self.data.shape[0]) * treatment3
        else:
            self.treatment2 = torch.rand(self.data.shape[0]) * max_angle
            self.treatment3 = torch.rand(self.data.shape[0]) * max_angle
        
        self.to_tensor = transforms.ToTensor()

        self.noise_std = noise_std
        self.non_additive_noise = non_additive_noise
        self.ite_mode = ite_mode

    def __getitem__(self,idx):
        img, target = self.data[idx], int(self.targets[idx])
        
        img_og = Image.fromarray(img.numpy())

        img = color_grayscale_arr(img, color = self.categorical_noise[idx])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())#, mode='L')
        
        if self.non_additive_noise:
            blurrer = transforms.GaussianBlur(kernel_size = (5,5), sigma= (self.noise_std))
            img = blurrer(img)

        
        img_new = transforms.functional.rotate(img, angle = self.treatment2[idx].item())
        img = transforms.functional.rotate(img, angle = self.treatment[idx].item())
        if self.ite_mode:
            img_new2 = transforms.functional.rotate(img, angle = self.treatment2[idx].item()) 
            if self.transform is not None:
                img_new2 = self.transform(img_new2)
            img_new2 = self.to_tensor(img_new2)

        if self.transform is not None:
            img = self.transform(img)
            img_new = self.transform(img_new)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img_og = self.to_tensor(img_og)
        img = self.to_tensor(img)
        img_new = self.to_tensor(img_new)
       
        if self.ite_mode:
            return img_og, img, self.categorical_noise[idx], self.treatment[idx], self.treatment2[idx], img_new, self.treatment3[idx], img_new2
        else:
            return img_og, img, self.categorical_noise[idx], self.treatment[idx], self.treatment2[idx], img_new

    
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, random_seed, colored = False, num_workers = 4, max_angle = 30, num_classes = 6, non_additive_noise = False, noise_std = 0., w_confounding = 0, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = random_seed
        self.num_workers = num_workers
        self.train_shuffle = True

        self.colored = colored
        self.max_angle = max_angle

        self.num_classes = num_classes #number of colors

        self.non_additive_noise = non_additive_noise
        self.noise_std = noise_std
        self.w_confounding = w_confounding


    def prepare_data(self, ite_mode = False, treatment2 = None, treatment3 = None):

        if self.colored:
            dataset = ColoredMNIST(root=os.path.join(DATA_DIR,"playground"), max_angle = self.max_angle, num_classes = self.num_classes, seed = self.seed, non_additive_noise  = self.non_additive_noise, noise_std = self.noise_std, w_confounding = self.w_confounding)
        else:
            def target_trans(x):
                return torch.nn.functional.one_hot(torch.LongTensor([x]),10)[0].float()
            dataset = MNIST(os.path.join(DATA_DIR,'playground'), train=True, transform=transforms.ToTensor(), download=True, target_transform = target_trans)
        
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

        self.conditional_dim = 10
    
        if self.colored:
            self.cf_dataset = ColoredMNISTCF(root=os.path.join(DATA_DIR,"playground"), max_angle = self.max_angle, num_classes = self.num_classes, seed = self.seed + 100, noise_std = self.noise_std, non_additive_noise = self.non_additive_noise, w_confounding = self.w_confounding, ite_mode = ite_mode, treatment2 = treatment2, treatment3 = treatment3)
        else:
            self.cf_dataset = None

        self.input_dim = 1
        self.output_dim = 3
        self.conditional_len = dataset.data.shape[-1]

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
        parser.add_argument('--num_classes', type=int, default=6)
        parser.add_argument('--colored', type=str2bool, default=True, help = "If true, uses the colored - rotated MNIST version")
        parser.add_argument('--non_additive_noise', type=str2bool, default=False, help = "If true, adds non-additive noise to the data")
        parser.add_argument('--noise_std', type=float, default=0.)
        parser.add_argument('--w_confounding', type=float, default=0., help = "Strength of confounding between Z and X.")
        return parser

if __name__ == "__main__":
    dataset = ColoredMNIST(root='./playground')
    img, y, color = dataset[0]
