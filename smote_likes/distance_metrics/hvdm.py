#!/usr/bin/env python3

r"""HVDM distance metric

    @Article{Wilson1997,
    author    = {D. R. Wilson and T. R. Martinez},
    title     = {Improved Heterogeneous Distance Functions},
    journal   = {Journal of Artificial Intelligence Research},
    year      = {1997},
    volume    = {6},
    pages     = {1--34},
    month     = {jan},
    doi       = {10.1613/jair.346},
    publisher = {{AI} Access Foundation},
    }

    hvdm(x,y) = \sqrt_{\sum_{a=1} ^ {m} d_{a}^2(x_{a}, y_{a})}
        where a E m is feature a

    hvdm(x,y) = \sqrt_{\sum_{a=1} ^ {num} d_{a}^2(x_{a}, y_{a}) + \sum_{b=1} ^ {cat} d_{b}^2(x_{b}, y_{b})}
        where a E num is a numerical feature and b E cat is a categorical feature

    d_{a}^2(x_{a}, y_{a}) = \left (   \frac {|x_{a} - y_{a}|} {4\sigma_a}   \right ) ^2
        d_{b}^2(x_{b}, y_{b}) = \sqrt {\sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 }

"""

# Authors: Lyubomir Danov <->
# License: -

import numpy
from .nvdm2 import normalized_vdm_2
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import DistanceMetric


def _split_arrays(arr, ind_cat_cols):
    ind_num_cols = [i for i in range(arr.shape[1]) if i not in ind_cat_cols]

    arr_cat = arr[:, ind_cat_cols]
    arr_num = arr[:, ind_num_cols]

    return arr_num, arr_cat


def hvdm(X: numpy.ndarray, target, ind_cat_cols: list = []):
    '''
    Computes HVDM with normalized_vdm2 for categorical data
    Currently does not set distance for missing values to 0
    Currently is only one target is given, all categorical distances are 0
    '''
    # if y is None:
    only_x = True

    x_num, x_cat = _split_arrays(X, ind_cat_cols)
    x_num_sd = numpy.std(x_num, axis=0, keepdims=True)
    x_num_normalzied = numpy.divide(x_num, (4*x_num_sd))

    if only_x:
        # square result s.t. it can be summed with cat_dist;
        # after that the sqrt is taken
        x_num_sqrt_dist = pairwise_distances(
            X=x_num_normalzied,
            metric='euclidean'
        )
        x_num_dist = numpy.square(x_num_sqrt_dist)
        x_cat_dist = normalized_vdm_2(x_cat, target)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist
