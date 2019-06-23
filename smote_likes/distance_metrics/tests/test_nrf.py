from ..nrf import NearestReferenceFeatures
import numpy


def test_NearestReferenceFeatures_bigger():
    # numpy.random.RandomState(12).randint(0,100,size=(5, 3))
    X_reference = numpy.array([
        [11,  6, 17],
        [2,  3,  3],
        [12, 16, 17],
        [5, 13,  2],
        [11, 10,  0]
    ])

    # numpy.random.RandomState(2).randint(0,100,size=(3, 3))
    X_interest = numpy.array([
        [8, 15, 13],
        [1, 11, 18],
        [11,  8,  7]
    ])

    exp_X_bigger = numpy.array([
        [11, 16, 17],
        [2, 13, 18],
        [12, 10, 17]
    ])

    X_bigger = NearestReferenceFeatures()._closest_nonclass(
        X_interest, X_reference, comparison_type='bigger')
    assert numpy.allclose(X_bigger, exp_X_bigger)


def test_NearestReferenceFeatures_smaller():
    # numpy.random.RandomState(12).randint(0,100,size=(5, 3))
    X_reference = numpy.array([
        [11,  6, 17],
        [2,  3,  3],
        [12, 16, 17],
        [5, 13,  2],
        [11, 10,  0]
    ])

    # numpy.random.RandomState(2).randint(0,100,size=(3, 3))
    X_interest = numpy.array([
        [8, 15, 13],
        [1, 11, 18],
        [11,  8,  7]
    ])

    exp_X_smaller = numpy.array([
        [5, 13, 3],
        [1, 10, 17],
        [5, 6, 3]
    ])

    X_smaller = NearestReferenceFeatures()._closest_nonclass(
        X_interest, X_reference, comparison_type='smaller')
    assert numpy.allclose(X_smaller, exp_X_smaller)
