#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy
import itertools
from .nvdm2 import _get_cond_proba

def nvdm_2_poc(X, target):
    cat_dicts = []
    for cat in range(X.shape[1]):
        cat_dicts.append(_get_cond_proba(X[:, cat], target))
        
    D = numpy.empty((X.shape[0], X.shape[0]))
    for i1, i2 in itertools.product(range(X.shape[0]), range(X.shape[0])):
        if i1 == i2:
            D[i1, i2] = 0
        else:
            D[i1, i2] = _nvdm2(X[i1, :], X[i2, :], cat_dicts)
        D[i2, i1] = D[i1, i2]

    return D

def _nvdm2(X, Y, cat_dicts):
    d = 0
    for a, b, search in zip(X, Y, cat_dicts):
        for trgt in search.keys():
            d += numpy.square(search[trgt][a] - search[trgt][b])
    return d
