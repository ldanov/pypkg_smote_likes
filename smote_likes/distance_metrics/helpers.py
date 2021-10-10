#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy


class ShapeError(IndexError):
    pass


def _split_arrays(arr, ind_cat_cols):
    ind_num_cols = [i for i in range(arr.shape[1]) if i not in ind_cat_cols]

    arr_cat = arr[:, ind_cat_cols]
    arr_num = arr[:, ind_num_cols]

    return arr_num, arr_cat


def _get_cond_proba(attrib: numpy.ndarray, y: numpy.ndarray) -> dict:
    """Calculate the conditional probability :math:`P_{a,x,c}`

    Parameters
    ----------
    attrib : numpy.ndarray
        Column vector of the attribute
    y : numpy.ndarray
        Column vector of the target variable

    Returns
    -------
    dict
        A dictionary with following structure:

        `target value -> attribute value -> attribute count`
    """
    attrib_count_class = _get_attrib_count_class(attrib, y)
    attrib_count_total = _get_attrib_count_total(attrib)
    target_uniques = numpy.unique(y)
    attrib_uniques = numpy.unique(attrib)
    cond_proba = _permutation_dict(target_uniques, attrib_uniques, 0)

    # all existing attribute occurances are always
    # part of attrib_count_total
    for attr_key, N_ax in attrib_count_total.items():
        for trgt in attrib_count_class.keys():
            N_axc = attrib_count_class[trgt].get(attr_key, 0)
            value = numpy.divide(N_axc, N_ax)
            cond_proba[trgt][attr_key] = value

    return cond_proba


def _get_attrib_count_class(attrib: numpy.ndarray, y: numpy.ndarray) -> dict:
    """Calculate the count :math:`N_{a,x,c}`

    Parameters
    ----------
    attrib : numpy.ndarray
        Column vector of the attribute
    y : numpy.ndarray
        Column vector of the target variable

    Returns
    -------
    dict
        A dictionary with following structure
            target value -> attribute value -> attribute count
    """
    assert attrib.shape == y.shape
    assert attrib.ndim == 1
    target_distincts = numpy.unique(y)
    class_count = {}

    for trgt in target_distincts:
        z = attrib[numpy.where(y == trgt, True, False)]
        class_count[trgt] = _get_count_dict(z)
    return class_count


def _get_attrib_count_total(attrib: numpy.ndarray) -> dict:
    """Calculate the count :math:`N_{a,x}`

    Parameters
    ----------
    attrib : numpy.ndarray
        Column vector of the attribute

    Returns
    -------
    dict
        A dictionary with structure
            attribute value: attribute count
    """
    assert attrib.ndim == 1
    total_count = _get_count_dict(attrib)

    return total_count


def _get_count_dict(vector: numpy.ndarray) -> dict:
    """Returns a dict with all unique values and their count.

    Parameters
    ----------
    vector : numpy.ndarray
        The vector of values to be counted

    Returns
    -------
    dict
        Keys are the unique values, values are the counts.
    """
    assert vector.ndim == 1
    value, count = numpy.unique(vector, return_counts=True)
    total_count = {}

    for i in range(value.shape[0]):
        total_count[value[i]] = count[i]
    return total_count


def _permutation_dict(keys0: list, keys1: list, value) -> dict:
    new_dict = {}
    for key0 in keys0:
        new_dict[key0] = {}
        for key1 in keys1:
            new_dict[key0][key1] = value
    return new_dict


def _remap_ndarray_dict(X, mapping):
    """ Values that are not part of mapping are set to 0.
    """
    new_X = numpy.zeros(X.shape)
    for key, value in mapping.items():
        new_X[X == key] = value
    return new_X


def _generate_interval_width(a, s):
    """ See pp 14 from Wilson, eq. (17), eq. (18)
    """
    max_a = numpy.max(a)
    min_a = numpy.min(a)
    w_a = numpy.abs(numpy.max(a) - numpy.min(a)) / s
    return (w_a, max_a, min_a)


def _get_all_interval_widths(X, s):
    """ Apply _generate_interval_width over all columns of X
    """
    return numpy.apply_along_axis(_generate_interval_width, 0, X, s=s)


def _discretize_column(x, s, w_a, max_a, min_a):
    """ See pp 14 from Wilson, eq. (18)
    """
    return numpy.where(x == max_a, s, numpy.floor((x-min_a)/w_a)+1)


def _get_interpolated_probability(P_Xu0c, P_Xu1c, z_X):
    """ See pp 17 from Wilson, eq. (23)
    """
    return P_Xu0c + numpy.multiply((P_Xu1c - P_Xu0c), z_X)


def _get_interpolated_location(x_a, u_xa, w_a, min_a):
    """ See pp 17 from Wilson, eq. (23), eq. (24)
    (x - mid_{a,u}) / (mid_{a,u+1} - mid_{a,u}) =
    (x - min_{a} - width_{a}*(u+0.5)) / (width_{a}*(u + 1.5 - u - 0.5)) =
    (x - min_{a} - width_{a}*(u+0.5)) / width_{a}*1 = 
    ((x - min_{a}) / width_{a}) - (u+0.5)
    """
    return ((x_a - min_a) / w_a) - (u_xa + 0.5)


def _mid_au(X_a, w_a, min_a) -> numpy.ndarray:
    """ See pp 17 from Wilson, eq. (24)
    """
    return (X_a + 0.5) * w_a + min_a


def _update_x_range(X_discrete, widths):
    """ See pp 17 from Wilson, eq. (24)
    "The value of u is found by first setting u = discretize_{a}(x), and then
    subtracting 1 from u if x < mid_{a,u}."
    """
    all_cols = []
    for col in range(X_discrete.shape[1]):
        z = _mid_au(X_discrete[:, col], widths[0, col], widths[2, col])
        all_cols.append(z)
    X_mids = numpy.stack(all_cols, 1)
    assert X_mids.shape == X_discrete.shape
    return numpy.where(X_discrete < X_mids, X_discrete - 1, X_discrete)
