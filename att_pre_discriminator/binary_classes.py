#!/usr/bin/python

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, data_file):
        self.labels     = labels
        self.list_IDs   = list_IDs
        self.data       = torch.Tensor(np.loadtxt(data_file))
        #print(self.data.size())
        #print("Data Loaded !")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self,index):
        ID = self.list_IDs[index]
        x = self.data[index]
        x = torch.Tensor(x)
        x = x/torch.norm(x)
        y = self.labels[ID]
        return x, y, ID


class PreDiscriminator(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(PreDiscriminator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.sigm = torch.nn.Sigmoid()
    
    def forward(self,x):
        x = self.sigm((self.linear1(x)))
        return x
