#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################

import argparse
import numpy as np

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Return the means and the standard deviations over the data dimension. To use as standardization parameters')
    parser.add_argument('data',help="data from which the means and the standard deviation vectors are computed", type=str)
    args = parser.parse_args()

    data = args.data
    data = np.loadtxt(data)

    mu = np.mean(data,axis=0)
    print(mu)
    std_dev = np.std(data,axis=0)
    std_dev[std_dev==0]=1
    
    np.save('mu',mu)
    np.save('std_dev',std_dev)
