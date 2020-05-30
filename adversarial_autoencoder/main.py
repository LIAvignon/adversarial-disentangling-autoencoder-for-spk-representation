#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################

import argparse
import numpy as np
import torch
import os
import sys
import math

sys.path.append('../utils/')

from scipy import signal
from torch.utils import data
from classes import Dataset, Autoencoder, Discrim, recons_loss_function, discrim_loss_function
from utils import create_partition_dict, create_labels_dict, txt_2_dict

# Here, attribute is binary

max_epochs = 100                    # Nb epochs

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run training, testing or generation')
    parser.add_argument('name_exp',help="forlder name of the experiment", type=str)
    parser.add_argument('xvectors', help="xvectors file", type=str)
    parser.add_argument('postprob', help="porsterior probabilities file", type=str)
    parser.add_argument('list_utt_id',help="list file of the utterances ID", type=str)
    parser.add_argument('att_labels_txt',help="Txt file containing all the spk id and the corresponding att label", type=str)
    args = parser.parse_args()

    #list_tr_id      = readListStrFile(args.list_tr_id)
    data_file       = args.xvectors
    prob_file       = args.postprob

    list_utt_id     = list(np.loadtxt(args.list_utt_id, dtype='str'))
    att_labels_txt  = args.att_labels_txt
    name_exp        = args.name_exp
    #lmax            = args.lmax

    #create output folder
    if not os.path.isdir(name_exp):os.mkdir(name_exp)
    if not os.path.isdir(name_exp+"/models"):os.mkdir(name_exp+"/models")

    #CUDA
    use_cuda        = torch.cuda.is_available()
    device          = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}

    #Dataset
    partition           = create_partition_dict(list_utt_id, [])
    spk_id_att_labels   = txt_2_dict(att_labels_txt)
    labels              = create_labels_dict(list_utt_id, [], spk_id_att_labels)

    #Generators
    training_set = Dataset(partition['train'], labels, data_file, prob_file)

    training_generator      = data.DataLoader(training_set, **params)

    # Layer dimension
    input_dim   = 512
    latent_dim  = 128

    input_dim_discrim   = latent_dim
    hidden_dim_discrim  = 64

    model_ae       = Autoencoder(input_dim, latent_dim)
    optimizer_ae   = torch.optim.SGD(model_ae.parameters(), lr = 0.0001, momentum=0.9)
    model_ae.to(device)

    model_discrim       = Discrim(input_dim_discrim, hidden_dim_discrim)
    optimizer_discrim   = torch.optim.SGD(model_discrim.parameters(), lr = 0.0001, momentum=0.9)
    model_discrim.to(device)

    # Training
    if True:
        f = open(name_exp+"/loss_"+name_exp+".txt","w+")
        f_recons = open(name_exp+"/recons_loss_"+name_exp+".txt","w+")
        f_ad = open(name_exp+"/ad_loss_"+name_exp+".txt","w+")
        f_discrim = open(name_exp+"/discrim_loss_"+name_exp+".txt","w+")


        for epoch in range(max_epochs):

            running_loss = 0.0
            running_recons_loss = 0.0
            running_ad_loss = 0.0
            running_discrim_loss = 0.0

            train_losses = []
            val_losses = []

            print("_____EPOCH: "+str(epoch+1)+'/'+str(max_epochs)+"_____")
            for i, data in enumerate(training_generator,0):
                local_batch, local_labels, local_probs = data[0].cuda(), data[1].cuda(), data[2].cuda()
                local_labels = local_labels.view(local_labels.size()[0],1).float()
                local_probs = local_probs.view(local_probs.size()[0],1).float()
                optimizer_ae.zero_grad()
                optimizer_discrim.zero_grad()

                outputs, z, w = model_ae(local_batch,local_probs)

                att_pred = model_discrim(z)

                loss_discrim = discrim_loss_function(att_pred,local_labels)
                loss_discrim.backward(retain_graph=True)
                optimizer_discrim.step()

                ad_loss = discrim_loss_function(1-att_pred,local_labels)
                recons_loss = recons_loss_function(outputs, local_batch)
                loss = recons_loss + ad_loss
                loss.backward()
                optimizer_ae.step()

                running_loss += loss.item()
                running_recons_loss += recons_loss.item()
                running_ad_loss += ad_loss.item()
                running_discrim_loss += loss_discrim.item()


                if i % 128 == 127:
                    print("..............")
                    print("Training Loss :         "+str(running_loss/(128)))
                    print("Training Recons Loss :  "+str(running_recons_loss/(128)))
                    print("Training Ad Loss :      "+str(running_ad_loss/(128)))
                    print("Training Discrim Loss : "+str(running_discrim_loss/(128)))

                    f.write(str(running_loss/(128))+"\n")
                    f_recons.write(str(running_recons_loss/(128))+"\n")
                    f_ad.write(str(running_ad_loss/(128))+"\n")
                    f_discrim.write(str(running_discrim_loss/(128))+"\n")

                    train_losses.append(running_loss/(128))
                    running_loss = 0.0
                    running_recons_loss = 0.0
                    running_ad_loss = 0.0
                    running_discrim_loss = 0.0


            epoch_tr_loss = np.array(train_losses).mean()

            f.flush()
            f_ad.flush()
            f_recons.flush()
            f_discrim.flush()

            torch.save(model_ae,name_exp+"/models/model_ae_"+str(epoch)+".pt")

        f.close()
        f_ad.close()
        f_recons.close()
        f_discrim.close()
        torch.save(model_discrim,name_exp+"/models/model_discrim_"+name_exp+".pt")

