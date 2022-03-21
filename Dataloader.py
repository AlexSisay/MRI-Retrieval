#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:36:32 2022

@author: alex
"""
import torch
#from torch import nn, optim
#from torchvision.transforms import Compose, ToTensor,Normalize
import nibabel as nb
import numpy as np # linear algebra
from torch.utils.data import Dataset, DataLoader


# Define a pytorch dataloader for this dataset
class ADNI_Dataset_hand(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images_list = [nb.load(image_path) for image_path in df['path']]
     
        #print(self.images.shape)
        #print(self.targets.shape)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        Xa = self.images_list[index]
        Xa = np.asarray(Xa.dataobj)
        #Xa = np.array(Xa.dataobj)
        Ya = torch.tensor(int(self.df['hand_type_idx'][index]))
        Ya = np.array(Ya)
        #Xa = Xa.reshape()
        #print('Type:',type(Xa), 'Shape:', Xa[0].shape)
        Xa = Xa.astype('float32')
        Xa /= 255                         #normalized to [0,1.0]
        Xa = np.expand_dims(Xa, axis = 1) #add channel=1 to the data, n * c * H * W
        Ya = Ya.astype('int64')
        
        X = torch.from_numpy(Xa)
        #print(X.shape)
        X = X.permute(1, 0, 2, 3)
        y = torch.from_numpy(Ya)
        if self.transform:
            X = self.transform(X)
        return X, y
# Define a pytorch dataloader for this dataset
class ADNI_Dataset_sex(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images_list = [nb.load(image_path) for image_path in df['path']]
     
        #print(self.images.shape)
        #print(self.targets.shape)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        Xa = self.images_list[index]
        Xa = np.asarray(Xa.dataobj)
        #Xa = np.array(Xa.dataobj)
        Ya = torch.tensor(int(self.df['gender_type_idx'][index]))
        Ya = np.array(Ya)
        #Xa = Xa.reshape()
        #print('Type:',type(Xa), 'Shape:', Xa[0].shape)
        Xa = Xa.astype('float32')
        Xa /= 255                         #normalized to [0,1.0]
        Xa = np.expand_dims(Xa, axis = 1) #add channel=1 to the data, n * c * H * W
        Ya = Ya.astype('int64')
        
        X = torch.from_numpy(Xa)
        #print(X.shape)
        X = X.permute(1, 0, 2, 3)
        y = torch.from_numpy(Ya)
        if self.transform:
            X = self.transform(X)
        return X, y
    