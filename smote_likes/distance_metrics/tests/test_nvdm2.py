# from ..nvdm2 import _get_cond_proba
# def test__get_cond_proba():
#     for dval in attrib_proba.values():
#         assert sum(dval.values()) == 1

import random
from string import ascii_lowercase

import numpy
import pandas
from sklearn.datasets import load_breast_cancer

from ..nvdm2 import normalized_vdm_2, get_cond_probas, nvdm2

def get_simple_test_data():
    X = numpy.array([[0,0], [0,1], [1,1]])
    y = numpy.array(["a", "b", "b"])
    return X, y

def _generate_cats(X, ncol):
    cat_vars = []
    for x in range(ncol):
        attrib = pandas.qcut(X[:,x], q=4, labels=False)
        cat_vars.append(attrib)
    cat_vars = numpy.array(cat_vars).transpose()
    assert cat_vars.shape == (X.shape[0], ncol)
    return cat_vars

def get_test_data_cats():
    X, y = load_breast_cancer(return_X_y=True)
    cat_vars = _generate_cats(X, 5)
    return cat_vars, y

def get_test_data_mixed():
    X, y = load_breast_cancer(return_X_y=True)
    cat_vars = _generate_cats(X, 5)
    all_vars = numpy.concatenate([cat_vars, X], axis=1)
    return all_vars, y


def test_nvdm2():
    X_cat, target = get_test_data_cats()
    cond_probas = get_cond_probas(X_cat, target)
    res = nvdm2(X_cat[0,], X_cat[1,], cond_proba_list=cond_probas)
    exp = numpy.array([0.0, 0.048373274112422864, 0.0, 0.0, 0.43370672772376007])
    assert numpy.allclose(res, exp)

def test_normalized_vdm_2():
    X_cat, target = get_test_data_cats()
    cond_probas = get_cond_probas(X_cat, target)
    exp = numpy.sum(nvdm2(X_cat[0,], X_cat[1,], cond_proba_list=cond_probas))
    all_dist = normalized_vdm_2(X_cat, target)
    assert all_dist[0,1] == exp

def test_cget_cond_probas_all_attribs_exist():
    # Given
    X_cat, target = get_test_data_cats()
    cond_probas = get_cond_probas(X_cat, target)
    unique_targets = numpy.unique(target)
    
    # Then
    for col in range(X_cat.shape[1]):
        exp = numpy.unique(X_cat[:,col])
        for value in exp:
            for trgt in unique_targets:
                assert cond_probas[col][trgt].get(value, None) is not None

def test_cget_cond_probas_prob_sum_to_1():
    # Given
    X_cat, target = get_test_data_cats()
    cond_probas = get_cond_probas(X_cat, target)
    unique_targets = numpy.unique(target)

    # Then
    for col in range(X_cat.shape[1]):
        exp = numpy.unique(X_cat[:,col])
        for value in exp:
            prob = 0
            for trgt in unique_targets:
                prob += cond_probas[col][trgt][value]
            assert prob == 1



# def test_hvdm_differences():
#     X, target = get_test_data_cats()
#     res_2 = normalized_vdm_2(X, target)
#     assert numpy.allclose(res_1, res_2)
