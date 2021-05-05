#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################
# Model classes. Here, the attribute is binary
#######################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import math
from scipy import signal

INPUT_SIZE      = 512               # Size of the x-vector

class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, data_file, prob_file):
        self.labels     = labels                                        # Hard labels
        self.list_IDs   = list_IDs                                      # Utterance Ids
        self.data       = torch.Tensor(np.loadtxt(data_file))           # x-vectors
        self.prob       = torch.Tensor(np.loadtxt(prob_file))           # Soft labels (posterior probability)
        print(self.data.size())
        print(self.prob.size())
        print("Data Loaded !")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self,index):
        ID  = self.list_IDs[index]                                      # Utterance Ids
        x   = self.data[index]                                          # x-vectors
        p   = self.prob[index]                                          # Soft labels
        x   = x/torch.norm(x)                                           # x-vector Length Normalisation
        y   = self.labels[ID]                                           # Hard labels
        return x, y, p, ID

class Autoencoder(nn.Module):

    def __init__(self,input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.input_dim  = input_dim
        self.linear1    = torch.nn.Linear(input_dim, latent_dim)
        self.linear2    = torch.nn.Linear(latent_dim+1, input_dim)
        self.bn1        = torch.nn.BatchNorm1d(num_features=latent_dim)

    def encode(self,x):                                                     # Encoder
        x = x/torch.norm(x,dim=1).view(x.size()[0],1)                       # x-vector Length Normalisation
        z = F.relu(self.linear1(x))
        z = self.bn1(z)
        return z

    def decode(self, z, attributes):                                        # Conditional Decoder
        x = torch.cat((z,attributes),1)
        x = torch.tanh(self.linear2(x))
        x = x/torch.norm(x,dim=1).view(x.size()[0],1)                       # Output x-vector Length Normalisation
        return x

    def forward(self, x, w, o='sr'):
        z = self.encode(x)
        if o!='sr':
            w = float(o)*torch.ones(z.size()[0]).cuda()
        w = w.reshape((z.size()[0],1))
        outputs = self.decode(z,w)
        return outputs, z, w


class Discrim(nn.Module):                                                   # Adversarial attribute discriminator
    def __init__(self,input_dim,hidden_dim):
        super(Discrim, self).__init__()
        self.input_dim  = input_dim
        self.linear1 = torch.nn.Linear(input_dim,hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim,1)

    def forward(self, x):
        att_pred = self.linear1(x)
        att_pred = F.relu(att_pred)
        att_pred = self.linear2(att_pred)
        att_pred = F.sigmoid(att_pred)
        return att_pred

def recons_loss_function(out_x, x):
    recons_loss = torch.mean(1-F.cosine_similarity(out_x, x.view(-1,INPUT_SIZE),dim=1))
    return recons_loss

def discrim_loss_function(pred, lbl):
    bce_loss        = torch.nn.BCELoss()
    discrim_loss    = bce_loss(pred, lbl)
    return discrim_loss
