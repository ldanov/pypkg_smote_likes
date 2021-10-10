#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy
from sklearn.metrics import euclidean_distances


def discretize_columns(X: numpy.ndarray, s: int) -> numpy.ndarray:
    r"""Transform a matrix of continuous into discrete features.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix with dimensions (observations, features).
        Note: should only contain continuous features.
    s : int
        Number of categories to group all elements of a feature into.
        Needs to be larger than 0.

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
    if not isinstance(s, int):
        t = type(s)
        raise ValueError("s needs to be int, but is {}".format(t))
    if not s > 0:
        raise ValueError("s needs to be larger than 0, but is {}".format(s))

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
        &= \sum_{a=1}^{num} \left (\frac {|x_{a} - y_{a}|} {4\sigma_a}\right ) ^2 \\
        & = \sum_{a=1}^{num} \left(\left| \frac {x_{a}} {4\sigma_a} - \frac {y_{a}} {4\sigma_a} \right|\right) ^2

    where `num` is the list of continuous attributes. Corresponds to 
    :math:`\sum_{a=1}^{num} d_{a}^2(x, y)` 
    from :py:func:`smote_likes.distance_metrics.hvdm`.

    """
    x_num_sd = numpy.std(X, axis=0, keepdims=True)
    dispersion = numpy.where(x_num_sd == 0, 1, 4*x_num_sd)

    x_num_normalzied = numpy.divide(X, dispersion)
    x_num_dist = euclidean_distances(
        X=x_num_normalzied,
        squared=True
    )
    return x_num_dist


def interpolated_vdm(X:numpy.ndarray, y:numpy.ndarray) -> numpy.ndarray:
    return NotImplementedError("wip")

def _generate_interval_width(a, s):
    """ See pp 14 from Wilson, eq. (17), eq. (18)
    """
    max_a = numpy.max(a)
    min_a = numpy.min(a)
    w_a = numpy.abs(numpy.max(a) - numpy.min(a)) / s
    return (w_a, max_a, min_a)


def _get_all_interval_widths(X, s, mid:bool = False):
    return numpy.apply_along_axis(_generate_interval_width, 0, X, s=s, mid=mid)


def _discretize_column(x, s, w_a, max_a, min_a):
    """ See pp 14 from Wilson, eq. (18)
    """
    return numpy.where(x == max_a, s, numpy.floor((x-min_a)/w_a)+1)


def _interpolated_probability_values(P_Xuc, P_Xu1c, z_X):
    """ See pp 17 from Wilson, eq. (23)
    """
    return P_Xuc + z_X * (P_Xuc - P_Xu1c)


def _midpoint_location(x_a, min_a, width_a, u_xa):
    """ See pp 17 from Wilson, eq. (24)
    """
    # TODO: make certain input is in matrix form
    z_x = ((x_a - min_a) / width_a) - (u_xa + 0.5)
    return z_x

# def _return_Pauc(X, y, )

def _mid_au(u, attrib_mins, attrib_widths):
    return attrib_mins + attrib_widths(u+0.5)

def get_x_mids(X_discrete, widths):
    return numpy.apply_along_axis(_mid_au, 0, X_discrete, widths[0,:], widths[2,:])

def get_x_range(X_discrete, X_mids):
    return numpy.where(X_discrete < X_mids, X_discrete - 1, X_discrete)