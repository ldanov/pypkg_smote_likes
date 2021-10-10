import numpy

from ..categorical import get_cond_probas, normalized_vdm_2, nvdm2
from .test_data import _get_test_data_cats


def test_nvdm2():
    # Given
    X_cat, target = _get_test_data_cats()

    # When
    cond_probas = get_cond_probas(X_cat, target)
    res = nvdm2(X_cat[0, ], X_cat[1, ], cond_proba_list=cond_probas)
    exp = numpy.array([0.0, 0.048373274112422864,
                      0.0, 0.0, 0.43370672772376007])

    # Then
    assert numpy.allclose(res, exp)


def test_normalized_vdm_2():
    # Given
    X_cat, target = _get_test_data_cats()

    # When
    cond_probas = get_cond_probas(X_cat, target)
    exp = numpy.sum(nvdm2(X_cat[0, ], X_cat[1, ], cond_proba_list=cond_probas))
    all_dist = normalized_vdm_2(X_cat, target)

    # Then
    assert numpy.allclose(all_dist[0, 1], exp)


def test_cget_cond_probas_all_attribs_exist():
    # Given
    X_cat, target = _get_test_data_cats()
    cond_probas = get_cond_probas(X_cat, target)
    unique_targets = numpy.unique(target)

    # Then
    for col in range(X_cat.shape[1]):
        exp = numpy.unique(X_cat[:, col])
        for value in exp:
            for trgt in unique_targets:
                assert cond_probas[col][trgt].get(value, None) is not None


def test_cget_cond_probas_prob_sum_to_1():
    # Given
    X_cat, target = _get_test_data_cats()
    cond_probas = get_cond_probas(X_cat, target)
    unique_targets = numpy.unique(target)

    # Then
    for col in range(X_cat.shape[1]):
        exp = numpy.unique(X_cat[:, col])
        for value in exp:
            prob = 0
            for trgt in unique_targets:
                prob += cond_probas[col][trgt][value]
            assert prob == 1
