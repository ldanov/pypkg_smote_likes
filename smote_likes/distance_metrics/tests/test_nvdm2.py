# from ..nvdm2 import _get_cond_proba
# def test__get_cond_proba():
#     for dval in attrib_proba.values():
#         assert sum(dval.values()) == 1

import random
from string import ascii_lowercase

import numpy
import pandas
from sklearn.datasets import load_breast_cancer

from ..nvdm2 import normalized_vdm_2, normalized_vdm_2_alt


def get_test_data_cats():
    X, y = load_breast_cancer(return_X_y=True)
    df = pandas.concat(
        [pandas.DataFrame(y, columns=['target']), pandas.DataFrame(X)], axis=1)
    cat_vars = []
    for x in range(5):
        random.seed(a=x)
        rand_position = random.sample(range(len(ascii_lowercase)), 4)
        labels = [ascii_lowercase[int(x)] for x in rand_position]
        attrib = pandas.qcut(df[x], q=4, labels=labels).values
        cat_vars.append(attrib)

    target = y
    test_cat = numpy.array(cat_vars).transpose()
    assert test_cat.shape == (X.shape[0], 5)
    return test_cat, target


def test_hvdm_differences():
    X, target = get_test_data_cats()
    res_1 = normalized_vdm_2_alt(X, target)
    res_2 = normalized_vdm_2(X, target)
    assert numpy.allclose(res_1, res_2)
