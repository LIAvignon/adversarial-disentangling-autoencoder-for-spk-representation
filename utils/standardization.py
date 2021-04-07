#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################

import argparse
import numpy as np

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Data standardization. Both train and test data are standardize')
    parser.add_argument('data_train',help="data from which the standardisation parameters are computed", type=str)
    parser.add_argument('data_test',help="data to standardize", type=str)
    args = parser.parse_args()

    data_tr = args.data_train
    data_tr = np.loadtxt(data_tr)
    data_te = args.data_test
    data_te = np.loadtxt(data_te)

    mu = np.mean(data_tr,axis=0)
    print(mu)
    std_dev = np.std(data_tr,axis=0)
    std_dev[std_dev==0]=1

    data_tr = data_tr - mu
    data_tr = data_tr/std_dev

    data_te = data_te - mu
    data_te = data_te/std_dev

    np.savetxt(args.data_train+".standardized",data_tr)
    np.savetxt(args.data_test+".standardized",data_te)

