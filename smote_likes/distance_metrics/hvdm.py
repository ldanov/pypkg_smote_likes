#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import DistanceMetric

from .nvdm2 import normalized_vdm_2


def _split_arrays(arr, ind_cat_cols):
    ind_num_cols = [i for i in range(arr.shape[1]) if i not in ind_cat_cols]

    arr_cat = arr[:, ind_cat_cols]
    arr_num = arr[:, ind_num_cols]

    return arr_num, arr_cat


def hvdm(X: numpy.ndarray, target, ind_cat_cols: list = []):
    r"""Computes HVDM distance metric with normalized_vdm2 for categorical data

    Parameters
    ----------
    X : numpy.ndarray
        [description]
    target : [type]
        [description]
    ind_cat_cols : list, optional
        [description], by default []

    Returns
    -------
    numpy.ndarray
        [description]
        Currently does not set distance for missing values to 0. 
        Currently is only one target is given, all categorical distances are 0. 


    Notes
    -----
    Generalised Harmonious Value Difference Metric (HVDM)  see :cite:t:`Wilson1997`

    .. math:: hvdm(x,y) = \sqrt { \sum_{a=1} ^ {m} d_{a}^2(x_{a}, y_{a}) }  
        = \sqrt {\sum_{a=1} ^ {num} d_{a}^2(x_{a}, y_{a}) + \sum_{b=1} ^ {cat} d_{b}^2(x_{b}, y_{b})}  
    where m is the number of features or attributes.

    .. math:: normalized_diff_{a}^2(x_{a}, y_{a}) = \left (   \frac {|x_{a} - y_{a}|} {4\sigma_a}   \right ) ^2


    .. math:: d_{b}^2(x_{b}, y_{b}) = \sqrt {\sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 }

    """

    # if y is None:
    only_x = True

    x_num, x_cat = _split_arrays(X, ind_cat_cols)

    if only_x:
        # square result s.t. it can be summed with cat_dist;
        # after that the sqrt is taken

        x_num_dist = normalized_diff(x_num)
        x_cat_dist = normalized_vdm_2(x_cat, target)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist


def normalized_diff(X: numpy.ndarray, y: numpy.ndarray):
    """Computes Normalised Difference Metric between continuous features


    Parameters
    ----------
    X : numpy.ndarray
        [description]

    Returns
    -------
    numpy.ndarray
        A distance matrix of size (X_rows, X_rows)
    """
    x_num_sd = numpy.std(x_num, axis=0, keepdims=True)
    x_num_normalzied = numpy.divide(x_num, (4*x_num_sd))
    x_num_sqrt_dist = pairwise_distances(
        X=x_num_normalzied,
        metric='euclidean',
        squared=True
    )
    # x_num_dist = numpy.square(x_num_sqrt_dist)
    return x_num_dist
