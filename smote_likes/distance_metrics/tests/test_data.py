import numpy
import pandas
from sklearn.datasets import load_breast_cancer


def _get_simple_test_data():
    X = numpy.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 3], [2, 4]])
    return X


def _generate_cats(X, ncol):
    cat_vars = []
    for x in range(ncol):
        attrib = pandas.qcut(X[:, x], q=4, labels=False)
        cat_vars.append(attrib)
    cat_vars = numpy.array(cat_vars).transpose()
    assert cat_vars.shape == (X.shape[0], ncol)
    return cat_vars


def _get_test_data_cats():
    X, y = load_breast_cancer(return_X_y=True)
    cat_vars = _generate_cats(X, 5)
    return cat_vars, y


def _get_test_data_mixed():
    X, y = load_breast_cancer(return_X_y=True)
    cat_vars = _generate_cats(X, 5)
    all_vars = numpy.concatenate([cat_vars, X], axis=1)
    return all_vars, y


def _get_nrf_test_data():
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
    return X_reference, X_interest
