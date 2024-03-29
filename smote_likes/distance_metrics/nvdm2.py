#!/usr/bin/env python3

r""" Normalized Value Difference Metric

    \sum_{b=1}^{cat}d_{b}^2(x_{b}, y_{b}) = \\
    \sum_{b=1}^{cat} \left ( \sqrt{    \sum_{c=1}^{C} \left ( \left | \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}   \right |  \right ) ^2 } \right ) ^2 = \\
    \sum_{c=1}^{C} \sum_{b=1}^{cat}      \left (  \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}}  \right ) ^2  
        where b E cat is a categorical feature



"""

# Authors: Lyubomir Danov <->
# License: -

import numpy
from sklearn.metrics import pairwise_distances

def normalized_vdm_2(X, target):
    list_conditional_probs = []
    for cat in range(X.shape[1]):
        list_conditional_probs.append(_get_cond_proba(X[:, cat], target))
    nvdm2 = _convert_euclidean(X, list_conditional_probs, target)
    return nvdm2


def _convert_euclidean(X, list_cprobs, target):
    unique_targets = numpy.unique(target)
    
    list_pdist_per_class = []
    for trgt in unique_targets:
        all_cols = []
        for col in range(X.shape[1]):
            P_ac = numpy.vectorize(list_cprobs[col][trgt].__getitem__)(X[:, col])
            all_cols.append(P_ac)
        X_P_c = numpy.vstack(all_cols).transpose()
        assert X.shape == X_P_c.shape
        pdist_c = numpy.square(pairwise_distances(X_P_c, metric='euclidean'))
        list_pdist_per_class.append(pdist_c)

    return sum(list_pdist_per_class)


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
            attrib_cond_proba[trgt][attr_key] = numpy.divide(attr_cond_count, attrib_count_total[attr_key])
    return attrib_cond_proba


def _get_cond_proba(attrib, target):

    attrib_count_class = _get_attrib_count_class(attrib, target)
    attrib_count_total = _get_attrib_count_total(attrib)
    return _get_cond_proba_dict(attrib_count_class, attrib_count_total)
