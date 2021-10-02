import numpy
import pytest
from sklearn.datasets import load_breast_cancer

from ..continuous import discretize_columns, normalized_diff


def _get_simple_test_data_2():
    X = numpy.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 3], [2, 4]])
    return X


def test_discretize_columns():
    # Given
    X = _get_simple_test_data_2()

    # When
    exp_5 = numpy.array([[1., 1.], [1., 2.], [3., 2.],
                         [5., 3.], [5., 4.], [5., 5.]])
    res_5 = discretize_columns(X, s=5)

    exp_2 = numpy.array([[1., 1.], [1., 1.], [2., 1.],
                         [2., 2.], [2., 2.], [2., 2.]])
    res_2 = discretize_columns(X, s=2)

    # Then
    assert numpy.allclose(res_2, exp_2)
    assert numpy.allclose(res_5, exp_5)


def test_discretize_columns_type_errors():
    # Given
    X = _get_simple_test_data_2()

    # When
    S = [float(2), "string", "2", 2.]

    #  Then
    for s in S:
        with pytest.raises(ValueError, match=r".*needs to be int.*"):
            discretize_columns(X, s)


def test_discretize_columns_sign_error():
    # Given
    X = _get_simple_test_data_2()

    # When
    s = -1

    #  Then
    with pytest.raises(ValueError, match=r".*needs to be larger than 0.*"):
        discretize_columns(X, s)


def test_normalized_diff1():
    # Given
    X = _get_simple_test_data_2()

    # When
    exp = numpy.array([[0., 0.03461538, 0.11220159, 0.44880637,
                        0.62188329, 0.86419098],
                       [0.03461538, 0., 0.07758621,
                           0.34496021, 0.44880637, 0.62188329],
                       [0.11220159, 0.07758621, 0.,
                           0.11220159, 0.21604775, 0.38912467],
                       [0.44880637, 0.34496021, 0.11220159,
                           0., 0.03461538, 0.13846154],
                       [0.62188329, 0.44880637, 0.21604775,
                           0.03461538, 0., 0.03461538],
                       [0.86419098, 0.62188329, 0.38912467,
                       0.13846154, 0.03461538, 0.]])
    res = normalized_diff(X)
    # Then
    assert numpy.allclose(res, exp)


def test_normalized_diff2():
    # Given
    X = numpy.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    # When
    exp = numpy.array([[0.  , 0.25, 0.5 , 0.25],
       [0.25, 0.  , 0.25, 0.5 ],
       [0.5 , 0.25, 0.  , 0.25],
       [0.25, 0.5 , 0.25, 0.  ]])
    res = normalized_diff(X)
    # Then
    assert numpy.allclose(res, exp)
