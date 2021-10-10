import numpy

from ..vdm import dvdm, hvdm, ivdm
from .test_data import _get_test_data_mixed


def test_hvdm():
    # Given
    X, y = _get_test_data_mixed()

    # When
    hvdm_res = hvdm(X[:5, :10], y[:5], range(0, 5))
    hvdm_exp = numpy.array([[0., 0.71178539, 0.71668654, 1.09116154, 0.4585822],
                            [0.71178539, 0., 0.40521449, 1.38314851, 0.2963686],
                            [0.71668654, 0.40521449, 0., 1.12812839, 0.457434],
                            [1.09116154, 1.38314851, 1.12812839, 0., 1.32069617],
                            [0.4585822, 0.2963686, 0.457434, 1.32069617, 0.]])

    assert numpy.allclose(hvdm_res, hvdm_exp)


def test_dvdm():
    # Given
    X, y = _get_test_data_mixed()

    # When
    dvdm_res = dvdm(X[15:20, :10], y[15:20], range(0, 5))
    dvdm_exp = numpy.array([[0., 0.97182532, 1.05409255, 1.24721913, 2.22361068],
                            [0.97182532, 0., 1.43372088, 1.26929552, 2.],
                            [1.05409255, 1.43372088, 0., 0.66666667, 2.46080384],
                            [1.24721913, 1.26929552, 0.66666667, 0., 2.3687784],
                            [2.22361068, 2., 2.46080384, 2.3687784, 0.]])

    assert numpy.allclose(dvdm_res, dvdm_exp)


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


def test_ivdm():
    # Given
    s = 5
    X, y = _get_test_data_mixed()
    X = X[:5, :10]
    y = y[:5]
    cat_values = list(range(0, 5))

    # When
    ivdm_exp = numpy.array([[0., 1.47952242, 1.36297176, 1.94097834, 1.08947945],
                            [1.47952242, 0., 2.00997768, 2.4827524, 1.63925655],
                            [1.36297176, 2.00997768, 0., 1.89158874, 1.91128793],
                            [1.94097834, 2.4827524, 1.89158874, 0., 2.44577429],
                            [1.08947945, 1.63925655, 1.91128793, 2.44577429, 0.]])
    ivdm_res = ivdm(X, y, cat_values, use_s=s)

    # Then
    assert numpy.allclose(ivdm_exp, ivdm_res)
