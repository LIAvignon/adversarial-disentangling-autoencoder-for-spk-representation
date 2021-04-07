from numpy import array, zeros, ones, inf, log, exp, isscalar, arange, argsort, hstack, concatenate, minimum, column_stack, abs
from numpy.linalg import solve
from copy import deepcopy


__author__ = "Andreas Nautsch"
__email__ = "nautsch@eurecom.fr"
__coauthor__ = ["Jose Patino", "Natalia Tomashenko", "Junichi Yamagishi", "Paul-Gauthier Noé", "Jean-François Bonastre", "Massimiliano Todisco", "Nicholas Evans"]
__credits__ = ["Niko Brummer", "Daniel Ramos", "Edward de Villiers", "Anthony Larcher"]
__license__ = "LGPLv3"


def logit(p):
    """logit function.
    This is a one-to-one mapping from probability to log-odds.
    i.e. it maps the interval (0,1) to the real line.
    The inverse function is given by SIGMOID.

    log_odds = logit(p) = log(p/(1-p))

    :param p: the inumpyut value

    :return: logit(inumpyut)
    """
    p = array(p)
    lp = zeros(p.shape)
    f0 = p == 0
    f1 = p == 1
    f = (p > 0) & (p < 1)

    if lp.shape == ():
        if f:
            lp = log(p / (1 - p))
        elif f0:
            lp = -inf
        elif f1:
            lp = inf
    else:
        lp[f] = log(p[f] / (1 - p[f]))
        lp[f0] = -inf
        lp[f1] = inf
    return lp


def sigmoid(log_odds):
    """SIGMOID: Inverse of the logit function.
    This is a one-to-one mapping from log odds to probability.
    i.e. it maps the real line to the interval (0,1).

    p = sigmoid(log_odds)

    :param log_odds: the inumpyut value

    :return: sigmoid(inumpyut)
    """
    p = 1 / (1 + exp(-log_odds))
    return p


def cllr(tar_llrs, nontar_llrs):
    log2 = log(2)

    tar_posterior = sigmoid(tar_llrs)
    non_posterior = sigmoid(-nontar_llrs)
    if any(tar_posterior == 0) or any(non_posterior == 0):
        return inf

    c1 = (-log(tar_posterior)).mean() / log2
    c2 = (-log(non_posterior)).mean() / log2
    c = (c1 + c2) / 2
    return c


def min_cllr(tar_llrs, nontar_llrs, monotonicity_epsilon=1e-6):
    [tar,non] = optimal_llr(tar_llrs, nontar_llrs, laplace=False, monotonicity_epsilon=monotonicity_epsilon)
    cmin = cllr(tar, non)
    return cmin


def pavx(y):
    """PAV: Pool Adjacent Violators algorithm.
    Non-paramtetric optimization subject to monotonicity.

    ghat = pav(y)
    fits a vector ghat with nondecreasing components to the
    data vector y such that sum((y - ghat).^2) is minimal.
    (Pool-adjacent-violators algorithm).

    optional outputs:
            width: width of pav bins, from left to right
                    (the number of bins is data dependent)
            height: corresponding heights of bins (in increasing order)

    Author: This code is a simplified version of the 'IsoMeans.m' code
    made available by Lutz Duembgen at:
    http://www.imsv.unibe.ch/~duembgen/software

    :param y: inumpyut value
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Inumpyut array is empty'
    n = y.shape[0]

    index = zeros(n, dtype=int)
    length = zeros(n, dtype=int)

    ghat = zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j + 1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = deepcopy(ghat[:ci + 1])
    width = deepcopy(length[:ci + 1])

    while n >= 0:
        for j in range(index[ci], n + 1):
            ghat[j - 1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def fast_actDCF(tar,non,plo,normalize=False):
    D = 1
    if not isscalar(plo):
        D = len(plo)
    T = len(tar)
    N = len(non)

    ii = argsort(hstack([-plo,tar]))
    r = zeros(T+D)
    r[ii] = arange(T+D) + 1
    r = r[:D]
    Pmiss = r - arange(start=D, step=-1, stop=0)

    ii = argsort(hstack([-plo,non]))  # -plo are thresholds
    r = zeros(N+D)
    r[ii] = arange(N+D) + 1
    r = r[:D]  # rank of thresholds
    Pfa = N - r + arange(start=D, step=-1, stop=0)

    Pmiss = Pmiss / T
    Pfa = Pfa / N

    Ptar = sigmoid(plo)
    Pnon = sigmoid(-plo)
    dcf = Ptar * Pmiss + Pnon * Pfa

    if normalize:
        dcf /= minimum([Ptar, Pnon])

    return dcf


def rocch_pava(tar_scores, nontar_scores, laplace=False):
    """ROCCH: ROC Convex Hull.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.

    :param tar_scores: vector of target scores
    :param nontar_scores: vector of classB_scores-target scores

    :return: a tupple of two vectors: Pmiss, Pfa
    """
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = concatenate((tar_scores, nontar_scores))
    # Pideal is the ideal, but classB_scores-monotonic posterior
    Pideal = concatenate((ones(Nt), zeros(Nn)))

    # It is important here that scores that are the same
    # (i.e. already in order) should NOT be swapped.rb
    perturb = argsort(scores, kind='mergesort')
    #
    Pideal = Pideal[perturb]

    if laplace:
       Pideal = hstack([1,0,Pideal,1,0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
      Popt = Popt[2:len(Popt)-2]

    nbins = width.shape[0]
    pmiss = zeros(nbins + 1)
    pfa = zeros(nbins + 1)

    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = Nn
    miss = 0

    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = Pideal[:left].sum()
        fa = N - left - Pideal[left:].sum()

    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn

    return pmiss, pfa, Popt, perturb


def optimal_llr_from_Popt(Popt, perturb, Ntar, Nnon, monotonicity_epsilon=1e-6):
    posterior_log_odds = logit(Popt)
    log_prior_odds = log(Ntar/Nnon)
    llrs = posterior_log_odds - log_prior_odds
    N = Ntar + Nnon
    llrs = llrs + arange(N) * monotonicity_epsilon/N # preserve monotonicity

    idx_reverse = zeros(N, dtype=int)
    idx_reverse[perturb] = arange(N)
    llrs_reverse = llrs[idx_reverse]
    tar_llrs = llrs_reverse[:Ntar]
    nontar_llrs = llrs_reverse[Ntar:]

    return tar_llrs, nontar_llrs


def optimal_llr(tar, non, laplace=False, monotonicity_epsilon=1e-6, compute_eer=False):
    # flag Laplace: avoids infinite LLR magnitudes;
    # also, this stops DET cureves from 'curling' to the axes on sparse data (DETs stay in more populated regions)
    scores = concatenate([non, tar])
    Pideal = concatenate([zeros(len(non)), ones(len(tar))])

    perturb = argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    if laplace:
        Pideal = hstack([1, 0, Pideal, 1, 0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
        Pideal = Pideal[2:len(Pideal) - 2]
        Popt = Popt[2:len(Popt) - 2]
        width[0] -= 2
        width[-1] -= 2

    posterior_log_odds = logit(Popt)
    log_prior_odds = log(len(tar) / len(non))
    llrs = posterior_log_odds - log_prior_odds
    N = len(tar) + len(non)
    llrs = llrs + arange(N) * monotonicity_epsilon / N  # preserve monotonicity

    idx_reverse = zeros(len(scores), dtype=int)
    idx_reverse[perturb] = arange(len(scores))
    tar_llrs = llrs[idx_reverse][len(non):]
    nontar_llrs = llrs[idx_reverse][:len(non)]

    if not compute_eer:
        return tar_llrs, nontar_llrs

    nbins = width.shape[0]
    pmiss = zeros(nbins + 1)
    pfa = zeros(nbins + 1)
    #
    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = non.shape[0]
    miss = 0
    #
    for i in range(nbins):
        if width[i] == 0:
            pass
        pmiss[i] = miss / len(tar)
        pfa[i] = fa /len(non)
        left = int(left + width[i])
        miss = Pideal[:left].sum()
        fa = len(tar) + len(non) - left - Pideal[left:].sum()
    #
    pmiss[nbins] = miss / len(tar)
    pfa[nbins] = fa / len(non)

    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = column_stack((xx, yy))
        dd = array([1, -1]) @ XY
        if minimum(*abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = solve(XY, array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (seg.sum())

        eer = max([eer, eerseg])

    return tar_llrs, nontar_llrs, eer



def ece(tar, non, plo):
    if isscalar(tar):
        tar = array([tar])
    if isscalar(non):
        non = array([non])
    if isscalar(plo):
        plo = array([plo])

    ece = zeros(plo.shape)
    for i, p in enumerate(plo):
        ece[i] = sigmoid(p) * (-log(sigmoid(tar + p))).mean()
        ece[i] += sigmoid(-p) * (-log(sigmoid(-non - p))).mean()

    ece /= log(2)

    return ece
