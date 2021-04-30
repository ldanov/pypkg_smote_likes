#!/usr/bin/env python3

# r""" Normalized Value Difference Metric

#     \sum_{b=1}^{cat}d_{b}^2(x_{b}, y_{b}) = \\
#     \sum_{b=1}^{cat} \left ( \sqrt{    \sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 } \right ) ^2 = \\
#     \sum_{c=1}^{C} \sum_{b=1}^{cat}      \left (  \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}  \right ) ^2
#         where b E cat is a categorical feature
# """

# Authors: Lyubomir Danov <->
# License: -

import numpy
import itertools

def normalized_vdm_2(X, target):
    r"""Computes Normalised Value Difference Metric 2

    Parameters
    ----------
    X : numpy.ndarray
        [description]
    target : numpy.ndarray
        The target class for each row in X.

    Returns
    -------
    numpy.ndarray
        [description]

    Notes
    -----
    Based on normalized_vdm2 (Equation 15) from :cite:t:`Wilson1997`. 
    As per the paper the square root is not taken, because the individual 
    attribute distances are themselves squared when used in the HVDM function.

    .. math:: 

        normalized\_vdm(x_{b}, y_{b}) = \sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 


    where C is the list of classes. 
    """


    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code
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


def _get_attrib_count_class(attrib, target):
    assert attrib.shape == target.shape
    assert attrib.ndim == 1
    target_distincts = numpy.unique(target)
    attribute_distincts = numpy.unique(attrib)
    class_count = {}

    # expand with all permutations of unique values
    for trgt in target_distincts:
        class_count[trgt] = {}
        for attr in attribute_distincts:
            class_count[trgt][attr] = 0

    for attr, trgt in zip(attrib, target):
        class_count[trgt][attr] += 1
    return class_count


def _get_attrib_count_total(attrib):
    assert attrib.ndim == 1

    total_count = {}
    for attr in attrib:
        if total_count.get(attr, None) is not None:
            total_count[attr] += 1
        else:
            total_count[attr] = 1

    return total_count


def _get_cond_proba_dict(attrib_count_class, attrib_count_total):
    attrib_cond_proba = {}
    for trgt, attr_dict in attrib_count_class.items():
        attrib_cond_proba[trgt] = {}
        for attr_key, attr_cond_count in attr_dict.items():
            attrib_cond_proba[trgt][attr_key] = numpy.divide(
                attr_cond_count, attrib_count_total[attr_key])
    return attrib_cond_proba


def _get_cond_proba(attrib, target):

    attrib_count_class = _get_attrib_count_class(attrib, target)
    attrib_count_total = _get_attrib_count_total(attrib)
    return _get_cond_proba_dict(attrib_count_class, attrib_count_total)
