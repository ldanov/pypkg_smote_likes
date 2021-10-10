#!/usr/bin/env python3

# Authors: Lyubomir Danov <->
# License: -

import numpy




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


class ShapeError(IndexError):
    pass
