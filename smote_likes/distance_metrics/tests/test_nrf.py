import numpy

from ..nrf import NearestReferenceFeatures
from .test_data import _get_nrf_test_data


def test_NearestReferenceFeatures_bigger():
    X_reference, X_interest = _get_nrf_test_data()

    exp_X_bigger = numpy.array([
        [11, 16, 17],
        [2, 13, 18],
        [12, 10, 17]
    ])

    X_bigger = NearestReferenceFeatures()._closest_nonclass(
        X_interest, X_reference, comparison_type='bigger')
    assert numpy.allclose(X_bigger, exp_X_bigger)


def test_NearestReferenceFeatures_smaller():
    X_reference, X_interest = _get_nrf_test_data()

    exp_X_smaller = numpy.array([
        [5, 13, 3],
        [1, 10, 17],
        [5, 6, 3]
    ])

    X_smaller = NearestReferenceFeatures()._closest_nonclass(
        X_interest, X_reference, comparison_type='smaller')
    assert numpy.allclose(X_smaller, exp_X_smaller)


def test_NearestReferenceFeatures_closest_nonclass():
    X_reference, X_interest = _get_nrf_test_data()

    exp_X_bigger = numpy.array([
        [11, 16, 17],
        [2, 13, 18],
        [12, 10, 17]
    ])
    exp_X_smaller = numpy.array([
        [5, 13, 3],
        [1, 10, 17],
        [5, 6, 3]
    ])

    X_bigger, X_smaller = NearestReferenceFeatures().closest_nonclass_values(
        X_interest, X_reference)

    assert numpy.allclose(X_smaller, exp_X_smaller)
    assert numpy.allclose(X_bigger, exp_X_bigger)
