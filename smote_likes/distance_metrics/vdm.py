#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy

from .nvdm2 import normalized_vdm_2
from .continuous import discretize_columns, normalized_diff


def hvdm(X: numpy.ndarray, y: numpy.ndarray, ind_cat_cols: list = None):
    r"""Computes HVDM distance metric with normalized_vdm_2 for categorical data
    and normalized_diff for numeric.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix with dimensions (observations, features).
    y : numpy.ndarray
        Target class for each row in X.
    ind_cat_cols : list, optional
        List of indices of categorical columns.
        If None assumes all are numeric, by default None.

    Returns
    -------
    numpy.ndarray
        Pair-wise distance matrix of dimensions (observations, observations).
        Currently does not set distance for missing values to 0 as per paper. 
        If only one target is given, all categorical distances are 0. See Notes for more information.

    Raises
    ------
    ValueError
        Raised when splitting X into two matrices (one for continuous and categorical each) fails.

    Notes
    -----
    Generalised Harmonious Value Difference Metric (HVDM)  see :cite:t:`Wilson1997`

    .. math:: 

        hvdm(x,y)  \\
        &= \sqrt { \sum_{f=1} ^ {F} d_{f}^2(x, y) } \\
        &= \sqrt {\sum_{a=1} ^ {num} d_{a}^2(x, y) + \sum_{b=1} ^ {cat} d_{b}^2(x, y)}  

    where `num` is the list of continuous attributes, `cat` is the list of categorical ones and
    `F` is the list of all attributes or features.

    .. math::

        d_{a}^2(x, y) \\
        &= normalized\_diff_a{^2}(x, y) \\
        &= \left (   \frac {|x_{a} - y_{a}|} {4\sigma_a}   \right ) ^2

    Implemented in :py:func:`smote_likes.distance_metrics.normalized_diff`

    .. math:: 

        d_{b}^2(x, y) \\
        &= normalized\_vdm_b{^2}(x, y) \\
        &= \sqrt {\sum_{c=1}^{C} \left | \frac {N_{b,x,c}} {N_{b,x}} - \frac {N_{b,y,c}} {N_{b,y}}   \right | ^2 } ^2

    where `C` is the list of classes or targets.
    Implemented in :py:func:`smote_likes.distance_metrics.normalized_vdm_2`

    Note: If only one target is given, all categorical distances are 0 because any :math:`\frac {N_{b,x,c}} {N_{b,x}} = 1`. 

    .. math:: 

        \sum_{a=1}^{A} \left | P_{x,a} - P_{y,a} \right | ^2 = \\
        \sqrt {\sum_{c=1}^{C} \left | \frac {N_{b,x,c}} {N_{b,x}} - \frac {N_{b,y,c}} {N_{b,y}}   \right | ^2 } ^2 = \\
        \sqrt {\sum_{}^{1} \left | 1 - 1 \right | ^2 } ^2

    """
    # TODO: handle missing values in x_num or x_cat
    if ind_cat_cols is None:
        ind_cat_cols = []

    n_obs = X.shape[0]

    if y is None:
        y = numpy.ones(shape=(n_obs,))

    x_num, x_cat = _split_arrays(X, ind_cat_cols)

    if x_num.size == 0 and x_cat.size == 0:
        raise ValueError("Splitting X into continuous \
            and discrete returned empty arrays.")

    if x_num.size == 0:
        x_num_dist = numpy.zeros(shape=(n_obs, n_obs))
    else:
        x_num_dist = normalized_diff(X=x_num)

    if x_cat.size == 0:
        x_cat_dist = numpy.zeros(shape=(n_obs, n_obs))
    else:
        x_cat_dist = normalized_vdm_2(X=x_cat, y=y)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist


def dvdm(X: numpy.ndarray, y: numpy.ndarray, ind_cat_cols: list = None, use_s: int = None):
    r"""Computes DVDM distance metric with normalized_vdm2 
    for all data after discretizing continuous data.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix with dimensions (observations, features).
    y : numpy.ndarray
        Target class for each row in X.
    ind_cat_cols : list, optional
        List of indices of categorical columns.
        If None assumes all are numeric, by default None.
    use_s: int, optional
        Number of discrete groups to be created for all continuous features.
        If none s will be the larger of 5 and number of classes in y. By default None.

    Returns
    -------
    numpy.ndarray
        Pair-wise distance matrix of dimensions (observations, observations).

    Raises
    ------
    ValueError
        Raised when splitting X into two matrices (one for continuous and categorical each) fails.

    Notes
    -----
    Discretized Value Difference Metric (HVDM)  see :cite:t:`Wilson1997`

    .. math:: 

        dvdm(x,y)  \\
        &= \sqrt { \sum_{b=1} ^ {cat} d_{b}^2(x, y)} 

    where `cat` is the list of categorical and numerical discretized to categorical 
    features. The algorithm used for discretization is :py:func:`smote_likes.distance_metrics.discretize_columns`.

    .. math::

        d_{b}^2(x, y) \\
        &= normalized\_vdm_b{^2}(x, y) \\
        &= \sqrt {\sum_{c=1}^{C} \left | \frac {N_{b,x,c}} {N_{b,x}} - \frac {N_{b,y,c}} {N_{b,y}}   \right | ^2 } ^2

    where `C` is the list of classes or targets.
    Implemented in :py:func:`smote_likes.distance_metrics.normalized_vdm_2`



    """
    if ind_cat_cols is None:
        ind_cat_cols = []

    n_obs = X.shape[0]

    if y is None:
        y = numpy.ones(shape=(n_obs,))

    x_num, x_cat = _split_arrays(X, ind_cat_cols)

    if x_num.size == 0 and x_cat.size == 0:
        raise ValueError("Splitting X into continuous \
            and discrete returned empty arrays.")

    if x_num.size == 0:
        x_num_dist = numpy.zeros(shape=(n_obs, n_obs))
    else:
        s = max(5, (numpy.unique(y)).shape[0]
                ) if use_s is None else use_s
        x_num_discrete = discretize_columns(X=x_num, s=s)
        x_num_dist = normalized_vdm_2(X=x_num_discrete, y=y)

    if x_cat.size == 0:
        x_cat_dist = numpy.zeros(shape=(n_obs, n_obs))
    else:
        x_cat_dist = normalized_vdm_2(X=x_cat, y=y)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist


def _split_arrays(arr, ind_cat_cols):
    ind_num_cols = [i for i in range(arr.shape[1]) if i not in ind_cat_cols]

    arr_cat = arr[:, ind_cat_cols]
    arr_num = arr[:, ind_num_cols]

    return arr_num, arr_cat
