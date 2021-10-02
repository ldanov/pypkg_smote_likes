import numpy
import pandas


def _generate_cats(X, ncol):
    cat_vars = []
    for x in range(ncol):
        attrib = pandas.qcut(X[:, x], q=4, labels=False)
        cat_vars.append(attrib)
    cat_vars = numpy.array(cat_vars).transpose()
    assert cat_vars.shape == (X.shape[0], ncol)
    return cat_vars
