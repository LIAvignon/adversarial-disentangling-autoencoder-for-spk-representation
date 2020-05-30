#!/usr/bin/python

from __future__ import division

import argparse
import numpy as np
import math
from sklearn import mixture


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Compute and print the means of a one dimentional N gaussian mixture')
    parser.add_argument('data',help="", type=str)
    parser.add_argument('N',help="number of mixture", type=int)
    args = parser.parse_args()

    data    = np.loadtxt(args.data)
    data    = data.reshape((len(data),1))
    print(data.shape)

    clf = mixture.GaussianMixture(n_components=args.N, covariance_type='full')
    clf.fit(data)
    print("Means")
    print(clf.means_)
