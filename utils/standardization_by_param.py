#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################

import argparse
import numpy as np

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Data standardization: data = (data-mu)/stdd')
    parser.add_argument('data',help="data to standardize", type=str)
    parser.add_argument('mu',help="numpy vector of means", type=str)
    parser.add_argument('stdd',help="numpy vector of standard deviations", type=str)
    args = parser.parse_args()

    data = args.data
    data = np.loadtxt(data)

    mu      = np.load(args.mu)
    stdd    = np.load(args.stdd)

    data = data - mu
    data = data/stdd

    np.savetxt(args.data+".standardized",data)

