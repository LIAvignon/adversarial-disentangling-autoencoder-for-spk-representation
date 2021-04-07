#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2020
#######################################################################

import argparse
import numpy as np

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Data destandardization: data = data*stdd + mu')
    parser.add_argument('data',help="data to destandardize", type=str)
    parser.add_argument('mu',help="numpy vector of means", type=str)
    parser.add_argument('stdd',help="numpy vector of strandard deviations", type=str)
    args = parser.parse_args()

    data = args.data
    data = np.loadtxt(data)

    mu      = np.load(args.mu)
    stdd    = np.load(args.stdd)

    data = data*stdd
    data = data + mu

    np.savetxt(args.data+".destandardized",data)

