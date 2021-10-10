import numpy
import pandas

from sklearn.datasets import load_breast_cancer

from ..helpers import _split_arrays


def _generate_cats(X, ncol):
    cat_vars = []
    for x in range(ncol):
        attrib = pandas.qcut(X[:, x], q=4, labels=False)
        cat_vars.append(attrib)
    cat_vars = numpy.array(cat_vars).transpose()
    assert cat_vars.shape == (X.shape[0], ncol)
    return cat_vars



def _get_test_data_mixed():
    X, y = load_breast_cancer(return_X_y=True)
    cat_vars = _generate_cats(X, 5)
    all_vars = numpy.concatenate([cat_vars, X], axis=1)
    return all_vars, y

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