#!/usr/bin/env python3

# r""" Normalized Value Difference Metric

#     \sum_{b=1}^{cat}d_{b}^2(x_{b}, y_{b}) = \\
#     \sum_{b=1}^{cat} \left ( \sqrt{    \sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 } \right ) ^2 = \\
#     \sum_{c=1}^{C} \sum_{b=1}^{cat}      \left (  \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}  \right ) ^2
#         where b E cat is a categorical feature
# """

# Authors: Lyubomir Danov <->
# License: -

import itertools

import numpy


def normalized_vdm_2(X, target, verbose:bool=False):
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

        &normalized\_vdm(x_{b}, y_{b}) \\
            &= \sum_{c=1}^{C} \left | P_{a,x,c} - P_{a,y,c} \right | ^2 \\
            &= \sum_{c=1}^{C} \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right | ^2



    where C is the list of classes. 
    """

    # TODO: check whether correct formula is shown in Notes
    # TODO: check whether formula shown matches code

    cond_proba_list = get_cond_probas(X=X, target=target)

    D = numpy.empty((X.shape[0], X.shape[0]))
    for i1, i2 in itertools.product(range(X.shape[0]), range(X.shape[0])):
        if verbose:
            print(i1, i2)
        if i1 == i2:
            D[i1, i2] = 0
            D[i2, i1] = 0
        elif i1 < i2:
            # if D[i1, i2] 
            distance_vector = nvdm2(X[i1, :], X[i2, :], cond_proba_list)
            D[i1, i2] = numpy.sum(distance_vector)
            D[i2, i1] = D[i1, i2]
        else:
            continue
    return D


def nvdm2(X: numpy.ndarray, Y: numpy.ndarray, cond_proba_list: list, idist: bool = True) -> list:
    r""" Calculate the nvdm2 between two observations 
    :math:`\sum_{c=1}^{C} \left | P_{a,x,c} - P_{a,y,c} \right |^2`
    for c classes and a attributes

    Parameters
    ----------
    X : numpy.ndarray
        The row-vector with all attributes of one observation
    Y : numpy.ndarray
        The row-vector with all attributes of another observation
    cond_proba_list : list of dict
        List of dicts of conditional probabilities
    idist : bool, optional
        If False returns the sum of all distances, if True the 
        individual distances between attraibutes. By default True.

    Returns
    -------
    list
        The individual distances between any two attributes of the observations.
    """
    dist_list = []
    for a, b, cond_pr_dict in zip(X, Y, cond_proba_list):
        d = 0
        for trgt in cond_pr_dict.keys():
            d += numpy.square(cond_pr_dict[trgt][a] - cond_pr_dict[trgt][b])
        dist_list.append(d)
    if not idist:
        dist_list = numpy.sum(dist_list)
    return dist_list


def get_cond_probas(X: numpy.ndarray, target: numpy.ndarray) -> list:
    """Get a list of all conditional probabilities in each column of X

    Parameters
    ----------
    X : numpy.ndarray
        The feature matrix with dimensions (observations, features).
    target : numpy.ndarray
        The target class for each row in X.

    Returns
    -------
    list
        Each entry contains the conditional probability :math:`P_{a,x,c}` 
        in a dictionary with following structure:
          
        `target value -> attribute value -> attribute count`
    """
    cond_proba_list = []
    if len(X.shape)!=2:
        raise ValueError("Shape of X needs to be 2 dimensional")
    for cat_ind in range(X.shape[1]):
        cond_proba_list.append(_get_cond_proba(X[:, cat_ind], target))
    
    return cond_proba_list


def _get_cond_proba(attrib: numpy.ndarray, target: numpy.ndarray) -> dict:
    """Calculate the conditional probability :math:`P_{a,x,c}`

    Parameters
    ----------
    attrib : numpy.ndarray
        The column vector of the attribute
    target : numpy.ndarray
        The column vector of the target variable

    Returns
    -------
    dict
        A dictionary with following structure:

        `target value -> attribute value -> attribute count`
    """
    attrib_count_class = _get_attrib_count_class(attrib, target)
    attrib_count_total = _get_attrib_count_total(attrib)
    target_uniques = numpy.unique(target)
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


def _get_attrib_count_class(attrib: numpy.ndarray, target: numpy.ndarray) -> dict:
    """Calculate the count :math:`N_{a,x,c}`

    Parameters
    ----------
    attrib : numpy.ndarray
        The column vector of the attribute
    target : numpy.ndarray
        The column vector of the target variable

    Returns
    -------
    dict
        A dictionary with following structure
            target value -> attribute value -> attribute count
    """
    assert attrib.shape == target.shape
    assert attrib.ndim == 1
    target_distincts = numpy.unique(target)
    class_count = {}

    for trgt in target_distincts:
        z = attrib[numpy.where(target == trgt, True, False)]
        class_count[trgt] = _get_count_dict(z)
    return class_count


def _get_attrib_count_total(attrib: numpy.ndarray) -> dict:
    """Calculate the count :math:`N_{a,x}`

    Parameters
    ----------
    attrib : numpy.ndarray
        The column vector of the attribute

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

def _permutation_dict(keys0: list, keys1:list, value) -> dict:
    new_dict = {}
    for key0 in keys0:
        new_dict[key0] = {}
        for key1 in keys1:
            new_dict[key0][key1] = value
    return new_dict