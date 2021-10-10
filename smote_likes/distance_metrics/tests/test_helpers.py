import numpy

from ..continuous import discretize_columns
from ..helpers import _get_all_interval_widths, _split_arrays, _update_x_range
from .test_data import _get_simple_test_data, _get_test_data_mixed


def test__split_arrays():
    # Given
    X, _ = _get_test_data_mixed()
    X = X[:3, :5]
    ind_cat_cols = [0, 1, 2]
    # When
    x_num, x_cat = _split_arrays(X, ind_cat_cols)
    x_cat_exp = numpy.array([[3, 0, 3], [3, 1, 3], [3, 2, 3]])
    x_num_exp = numpy.array([[3, 3], [3, 0], [3, 3]])

    # Then
    assert numpy.allclose(x_num, x_num_exp)
    assert numpy.allclose(x_cat, x_cat_exp)


def test__get_x_mids():
    # Given
    s = 3
    X = _get_simple_test_data()
    X_discrete = discretize_columns(X, s)
    widths = _get_all_interval_widths(X, s)

    # When
    exp = numpy.array([[1., 0.],
                       [1., 0.],
                       [2., 0.],
                       [3., 1.],
                       [3., 2.],
                       [3., 2.]])
    res = _update_x_range(X_discrete, widths)

    # Then
    assert numpy.allclose(res, exp)
