#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy
from sklearn.metrics import pairwise_distances

from .nvdm2 import normalized_vdm_2


def hvdm(X: numpy.ndarray, y, ind_cat_cols: list = None):
    r"""Computes HVDM distance metric with normalized_vdm2 for categorical data

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
        Currently does not set distance for missing values to 0. 
        TODO: set distance for missing values to 0.
        Currently if only one target is given, all categorical distances are 0. 
        TODO: handle only one class with :math:`\sum_{a=1}^{A} \left | P_{x,a} - P_{y,a} \right |^2`

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
        &= normalized\_diff_a(x, y) \\
        &= \left (   \frac {|x_{a} - y_{a}|} {4\sigma_a}   \right ) ^2

    Implemented in :py:func:`smote_likes.distance_metrics.normalized_diff`

    .. math:: 

        d_{b}^2(x, y) \\
        &= normalized\_vdm_b(x, y) \\
        &= \sqrt {\sum_{c=1}^{C} \left | \frac {N_{b,x,c}} {N_{b,x}} - \frac {N_{b,y,c}} {N_{b,y}}   \right | ^2 }^2

    where `C` is the list of classes or targets.
    Implemented in :py:func:`smote_likes.distance_metrics.normalized_vdm_2`

    """

    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
    # TODO: add tests including missing x_num or x_cat
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


def discretize_columns(X, s):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    s : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    # TODO: documentation
    # TODO: tests
    widths = _get_all_interval_widths(X, s)
    all_cols = []
    for col in range(X.shape[1]):
        z = _discretize_column(
            X[:, col], s, widths[0, col], widths[1, col], widths[2, col])
        all_cols.append(z)
    return numpy.stack(all_cols, 1)


def ivdm(X: numpy.ndarray, y: numpy.ndarray, ind_cat_cols: list = None):
    """[summary]

    Parameters
    ----------
    X : numpy.ndarray
        [description]
    y : numpy.ndarray
        [description]
    ind_cat_cols : list, optional
        [description], by default None

    Returns
    -------
    numpy.ndarray
        Pair-wise distance matrix of dimensions (observations, observations).

    Raises
    ------
    ValueError
        [description]
    """    
    # TODO: documentation
    # TODO: tests
    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
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
        s = max(5, (numpy.unique(y)).shape[0])
        x_num_discrete = discretize_columns(X=x_num, s=s)
        x_num_dist = normalized_vdm_2(X=x_num_discrete, y=y)

    if x_cat.size == 0:
        x_cat_dist = numpy.zeros(shape=(n_obs, n_obs))
    else:
        x_cat_dist = normalized_vdm_2(X=x_cat, y=y)

    x_dist = numpy.sqrt(x_num_dist + x_cat_dist)
    return x_dist


def normalized_diff(X: numpy.ndarray) -> numpy.ndarray:
    r"""Computes Normalised Difference Metric between continuous features

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix with dimensions (observations, features).

    Returns
    -------
    numpy.ndarray
        Pair-wise distance matrix of dimensions (observations, observations).

    Notes
    -----
    Based on normalized_vdm2 (Equation 15) from :cite:t:`Wilson1997`. 
    As per the paper the square root is not taken, because the individual 
    attribute distances are themselves squared when used in the HVDM function.

    .. math:: 
        normalized\_diff(x, y) \\
        &= \sum_{a=1}^{num} normalized\_diff_a(x, y) \\
        &= \sum_{a=1}^{num} \left (\frac {|x_{a} - y_{a}|} {4\sigma_a}\right ) ^2

    where `num` is the list of continuous attributes. Corresponds to 
    :math:`\sum_{a=1}^{num} d_{a}^2(x, y)` 
    from :py:func:`smote_likes.distance_metrics.hvdm`.

    """

    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
    x_num_sd = numpy.std(X, axis=0, keepdims=True)
    x_num_normalzied = numpy.divide(X, (4*x_num_sd))
    x_num_dist = pairwise_distances(
        X=x_num_normalzied,
        metric='euclidean',
        squared=True
    )
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


def _split_arrays(arr, ind_cat_cols):
    ind_num_cols = [i for i in range(arr.shape[1]) if i not in ind_cat_cols]

    arr_cat = arr[:, ind_cat_cols]
    arr_num = arr[:, ind_num_cols]

    return arr_num, arr_cat
