#!/usr/bin/python

import argparse
import numpy as np
import sys

sys.path.append('/local_disk/zephyr2/pgnoe/x_vector/adversarial-disentangling-autoencoder-for-spk-representation/zebra/')
from performance import *

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Calibrate score with PAV to obtain posterior probabilities')
    parser.add_argument('scores',help="scores to calibrate", type=str)
    parser.add_argument('labels',help="corresponding class label (0 or 1)", type=str)
    args = parser.parse_args()

    scores = np.loadtxt(args.scores)
    Pideal = np.loadtxt(args.labels)

    perturb = argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    l_tar = len(np.where(Pideal==1))
    l_non = len(np.where(Pideal==0))

    Popt, width, foo = pavx(Pideal)

    idx_reverse = zeros(len(scores), dtype=int)
    idx_reverse[perturb] = arange(len(scores))

    Popt = Popt[idx_reverse]
    np.savetxt(args.scores+".cal", Popt)

