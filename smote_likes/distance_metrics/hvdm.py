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


def hvdm(X: numpy.ndarray, target, ind_cat_cols: list = None):
    r"""Computes HVDM distance metric with normalized_vdm2 for categorical data

    Parameters
    ----------
    X : numpy.ndarray
        [description]
    target : [type]
        [description]
    ind_cat_cols : list, optional
        [description], by default None

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

    .. math:: d_{a}(x_{a}, y_{a}) = \left (   \frac {|x_{a} - y_{a}|} {4\sigma_a}   \right ) 

    Implemented in :py:func:`smote_likes.distance_metrics.normalized_diff`

    .. math:: d_{b}(x_{b}, y_{b}) = \sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 

    where C is the list of classes or targets.
    Implemented in :py:func:`smote_likes.distance_metrics.normalized_vdm_2`

    """

    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
    if ind_cat_cols is None:
        ind_cat_cols = []
    # if y is None:
    only_x = True

    x_num, x_cat = _split_arrays(X, ind_cat_cols)

    # TODO: more efficient if any of x_num, x_cat is empty
    if only_x:
        # square result s.t. it can be summed with cat_dist;
        # after that the sqrt is taken

        x_num_dist = normalized_diff(x_num)
        x_cat_dist = normalized_vdm_2(x_cat, target)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist


def normalized_diff(X: numpy.ndarray, y: numpy.ndarray):
    r"""Computes Normalised Difference Metric between continuous features

    Parameters
    ----------
    X : numpy.ndarray
        [description]

    Returns
    -------
    numpy.ndarray
        A distance matrix of size (X_rows, X_rows)

    Notes
    -----
    Based on normalized_vdm2 (Equation 15) from :cite:t:`Wilson1997`. 
    As per the paper the square root is not taken, because the individual 
    attribute distances are themselves squared when used in the HVDM function.

    .. math:: 
        d_{a}(x_{a}, y_{a}) = \left (   \frac {|x_{a} - y_{a}|} {4\sigma_a}   \right ) 

    """

    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
    x_num_sd = numpy.std(x_num, axis=0, keepdims=True)
    x_num_normalzied = numpy.divide(x_num, (4*x_num_sd))
    x_num_sqrt_dist = pairwise_distances(
        X=x_num_normalzied,
        metric='euclidean',
        squared=True
    )
    # x_num_dist = numpy.square(x_num_sqrt_dist)
    return x_num_dist


def _generate_interval_width(a, s):
    max_a = numpy.max(a)
    min_a = numpy.min(a)
    w_a = numpy.abs(numpy.max(a) - numpy.min(a)) / s
    return (w_a, max_a, min_a)


def _get_all_interval_widths(X, s):
    return numpy.apply_along_axis(_generate_interval_width, 0, X, s=s)


def _discretize_column(x, s, w_a, max_a, min_a):
    return numpy.where(x == max_a, s, numpy.ceil((x-min_a)/w_a)+1)


def discretize_columns(X, s):
    # TODO: documentation
    # TODO: tests
    widths = _get_all_interval_widths(X, s)
    all_cols = []
    for col in range(X.shape[1]):
        z = _discretize_column(
            X[:, col], s, widths[0, col], widths[1, col], widths[2, col])
        all_cols.append(z)
    return numpy.stack(all_cols, 1)


def ivdm(X: numpy.ndarray, target, ind_cat_cols: list = None):
    # TODO: documentation
    # TODO: tests
    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
    if ind_cat_cols is None:
        ind_cat_cols = []
    # if y is None:
    only_x = True

    x_num, x_cat = _split_arrays(X, ind_cat_cols)

    # TODO: more efficient if any of x_num, x_cat is empty
    if only_x:
        # square result s.t. it can be summed with cat_dist;
        # after that the sqrt is taken
        s = max(5, (numpy.unique(target)).shape[0])
        x_num_discrete = discretize_columns(X=x_num, s=s)
        x_num_dist = normalized_vdm_2(X=x_num_discrete, target=target)
        x_cat_dist = normalized_vdm_2(X=x_cat, target=target)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist
