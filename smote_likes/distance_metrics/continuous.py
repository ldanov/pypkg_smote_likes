#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy
from sklearn.metrics import pairwise_distances


def discretize_columns(X, s) -> numpy.ndarray:
    r"""Transform a matrix of continuous into discrete features.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix with dimensions (observations, features).
        Note: should only contain continuous features.
    s : int
        Number of categories to group all elements of a feature into.

    Returns
    -------
    numpy.ndarray
        Matrix of dimensions like original with values representing categories. 

    Notes
    -----
    Discretizing of continuous values see eq. 18, pp.14 :cite:t:`Wilson1997`

    .. math::

        discretize_a(x) = 
        \begin{cases} 
            s, \text{if $x=max_a$, else} \\
            \lfloor (x-min_a)/w_a \rfloor+1 
        \end{cases}

    """
    # TODO: tests
    widths = _get_all_interval_widths(X, s)
    all_cols = []
    for col in range(X.shape[1]):
        z = _discretize_column(
            X[:, col], s, widths[0, col], widths[1, col], widths[2, col])
        all_cols.append(z)
    return numpy.stack(all_cols, 1)


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
    Based on normalized_diff (Equation 13) from :cite:t:`Wilson1997`. 
    As per the paper the square root is not taken, because the individual 
    attribute distances are themselves squared when used in the HVDM function. 
    For overview see pp. 22. 

    .. math:: 
        normalized\_diff(x, y) \\
        &= \sum_{a=1}^{num} normalized\_diff_a(x, y) \\
        &= \sum_{a=1}^{num} \left (\frac {|x_{a} - y_{a}|} {4\sigma_a}\right ) ^2

    where `num` is the list of continuous attributes. Corresponds to 
    :math:`\sum_{a=1}^{num} d_{a}^2(x, y)` 
    from :py:func:`smote_likes.distance_metrics.hvdm`.

    """
    # Note: only basic tests as implementation heavily based on numpy
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
    """ See pp 14 from Wilson, eq. (17), eq. (18)
    """
    max_a = numpy.max(a)
    min_a = numpy.min(a)
    w_a = numpy.abs(numpy.max(a) - numpy.min(a)) / s
    return (w_a, max_a, min_a)


def _get_all_interval_widths(X, s):
    return numpy.apply_along_axis(_generate_interval_width, 0, X, s=s)


def _discretize_column(x, s, w_a, max_a, min_a):
    """ See pp 14 from Wilson, eq. (17), eq. (18)
    """
    return numpy.where(x == max_a, s, numpy.floor((x-min_a)/w_a)+1)
