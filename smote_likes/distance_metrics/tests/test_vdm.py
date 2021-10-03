import numpy
from sklearn.datasets import load_breast_cancer

from ..vdm import _split_arrays, dvdm, hvdm
from .test_helper import _generate_cats


def _get_test_data_mixed():
    X, y = load_breast_cancer(return_X_y=True)
    cat_vars = _generate_cats(X, 5)
    all_vars = numpy.concatenate([cat_vars, X], axis=1)
    return all_vars, y


def test_hvdm():
    # Given
    X, y = _get_test_data_mixed()

    # When
    hvdm_res = hvdm(X[:5, :10], y[:5], range(0,5))
    hvdm_exp = numpy.array([[0., 0.71178539, 0.71668654, 1.09116154, 0.4585822 ],
       [0.71178539, 0., 0.40521449, 1.38314851, 0.2963686 ],
       [0.71668654, 0.40521449, 0., 1.12812839, 0.457434  ],
       [1.09116154, 1.38314851, 1.12812839, 0., 1.32069617],
       [0.4585822, 0.2963686, 0.457434 , 1.32069617, 0. ]])

    assert numpy.allclose(hvdm_res, hvdm_exp)


def test_dvdm():
    # Given
    X, y = _get_test_data_mixed()

    # When
    dvdm_res = dvdm(X[15:20, :10], y[15:20], range(0,5))
    dvdm_exp = numpy.array([[0., 0.97182532, 1.05409255, 1.24721913, 2.22361068],
       [0.97182532, 0., 1.43372088, 1.26929552, 2. ],
       [1.05409255, 1.43372088, 0., 0.66666667, 2.46080384],
       [1.24721913, 1.26929552, 0.66666667, 0., 2.3687784 ],
       [2.22361068, 2., 2.46080384, 2.3687784, 0. ]])

    assert numpy.allclose(dvdm_res, dvdm_exp)


def test__split_arrays():
    # Given
    X, _ = _get_test_data_mixed()
    X = X[:3, :5]
    ind_cat_cols = [0, 1, 2]
    # When
    x_num, x_cat = _split_arrays(X, ind_cat_cols)
    x_cat_exp = numpy.array([[3,0,3], [3,1,3], [3,2,3]])
    x_num_exp = numpy.array([[3,3], [3,0], [3,3]])
    

    # Then
    assert numpy.allclose(x_num, x_num_exp)
    assert numpy.allclose(x_cat, x_cat_exp)
    # raise AssertionError("no test yet")


def test_compare_categorical():
    # Given
    # cat only dataset
    X_mixed, y = _get_test_data_mixed()
    X = X_mixed[:, :5]
    cat_col_ind = range(0, 5)

    # When
    res_hvdm = hvdm(X=X, y=y, ind_cat_cols=cat_col_ind)
    res_dvdm = dvdm(X=X, y=y, ind_cat_cols=cat_col_ind)

    # Then
    assert numpy.allclose(res_dvdm, res_hvdm)
