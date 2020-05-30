#!/usr/bin/python

import argparse
import numpy as np
import torch
import copy
import os
import sys

sys.path.append('../utils/')

from torch.utils import data
from binary_classes import Dataset, PreDiscriminator
from utils import create_partition_dict, create_labels_dict, txt_2_dict

#split = 0.2
N_epoch = 15

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('xvectors',help="xvectors file", type=str)
    parser.add_argument('list_utt_id',help="list file of the utterances ID", type=str)
    parser.add_argument('att_labels_txt',help="Txt file containing all the spk id and the corresponding attribute label", type=str)
    parser.add_argument('name_exp',help="name of the experiment", type=str)
    parser.add_argument('thres',help="classification decision threshold", type=float)
    parser.add_argument("-t", "--test", help="Testing",action="store_true")
    parser.add_argument("-g", "--gen", help="Generate the posterior probabilities", action="store_true")
    args = parser.parse_args()

    data_file       = args.xvectors
    list_utt_id     = list(np.loadtxt(args.list_utt_id, dtype='str'))
    att_labels_txt  = args.att_labels_txt
    name_exp        = args.name_exp

    #create output folder
    if not os.path.isdir(name_exp):os.mkdir(name_exp)

    #CUDA
    use_cuda        = torch.cuda.is_available()
    device          = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}
    max_epochs = N_epoch

    #Dataset
    partition = create_partition_dict(list_utt_id, list_utt_id)
    spk_id_att_labels = txt_2_dict(att_labels_txt)
    labels = create_labels_dict(list_utt_id,list_utt_id,spk_id_att_labels)

    #Generators
    training_set = Dataset(partition['train'], labels, data_file)
    #training_set, validation_set = data.random_split(training_set, [training_set.__len__()-int(training_set.__len__()*split), int(training_set.__len__()*split)])
    #training_set = Dataset(partition['train'], labels, data_file)
    #test_set = Dataset(partition['test'], labels, data_file)

    training_generator      = data.DataLoader(training_set, **params)
    #validation_generator    = data.DataLoader(validation_set, **params)
    #test_generator          = data.DataLoader(test_set, **params)


    input_dim   = 512
    output_dim  = 1


    #CLASSIFICATION MODEL
    model       = PreDiscriminator(input_dim, output_dim)

    criterion   = torch.nn.BCELoss()
    optimizer   = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    model.to(device)

    if not args.test and not args.gen:
        f = open(name_exp+"/loss_"+name_exp+".txt","w+")

        for epoch in range(max_epochs):

            running_loss = 0.0
            train_losses = []
            validation_losses = []
            print("_____EPOCH: "+str(epoch+1)+'/'+str(max_epochs)+"_____")
            for i, data in enumerate(training_generator,0):
                local_batch, local_labels = data[0].cuda(), data[1].cuda()
                local_labels = local_labels.view(local_labels.size()[0],1).float()

                optimizer.zero_grad()

                outputs = model(local_batch)

                loss = criterion(outputs, local_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 128 == 127:
                    print("Training Loss : "+str(running_loss/(128)))
                    train_losses.append(running_loss/(128))
                    running_loss = 0.0
            epoch_tr_loss = np.array(train_losses).mean()

            f.write(str(epoch_tr_loss)+"\n")
            f.flush()
        f.close()
        torch.save(model,name_exp+"/model_"+name_exp+".pt")

    if args.test:

        thres = args.thres

        model       = torch.load(name_exp+"/model_"+name_exp+".pt")
        correct     = 0
        total       = 0
        correct_0   = 0
        correct_1   = 0
        total_1     = 0
        with torch.no_grad():
            for data in test_generator:
                local_batch, local_labels = data[0].cuda(), data[1].cuda()
                local_labels = local_labels.view(local_labels.size()[0],1).float()

                outputs = model(local_batch)
                outputs[outputs>thres] = 1
                outputs[outputs<=thres] = 0

                for i in range(local_labels.size()[0]):
                    total += 1
                    if local_labels[i].item() == 1:
                        total_1 += 1
                    if outputs[i].item() == local_labels[i].item():
                        correct += 1
                    if outputs[i].item() == local_labels[i].item():
                        if outputs[i].item() == 1:
                            correct_1  += 1
                        if outputs[i].item() == 0:
                            correct_0    += 1

        print("Decision threshold: "+str(thres))
        print("Accuracy: "+str(correct/total))
        print("Accuracy over the class 1: "+str(correct_1/total_1))
        print("Accuract over the class 0: "+str(correct_0/(total-total_1)))

    if args.gen:

        model = torch.load(name_exp+"/model_"+name_exp+".pt")
        model.eval()
        
        data = torch.Tensor(np.loadtxt(args.xvectors))

        

        P = []
        with torch.no_grad():
            for i in range(data.size()[0]):
                d = data[i].view(1,input_dim).cuda()
                d = d/torch.norm(d)
                p = model(d)
                P.append(np.array(p.cpu().view(1)))

        P = np.array(P)
        P = P.reshape(len(P))

        np.savetxt(args.xvectors+".postprob",P)


