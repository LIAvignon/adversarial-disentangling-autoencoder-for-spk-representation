#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################

import numpy as np

def create_partition_dict(list_tr_id, list_te_id):
    partition               = dict()
    partition['train']      = list_tr_id
    partition['test']       = list_te_id
    return partition

def create_labels_dict(list_tr_id,list_te_id,spk_id_att_labels):
    labels_dict     = dict()
    list_id         = list_tr_id+list_te_id
    for i in range(len(list_id)):
        labels_dict[list_id[i]] = spk_id_att_labels[list_id[i].split('-')[0]]
    return labels_dict

def txt_2_dict(txt_file):
    d = dict()
    fh = open(txt_file)
    x = []
    for line in fh.readlines():
        y = [str(value) for value in line.split()]
        d[y[0]] = int(y[1])
    fh.close()
    return d

