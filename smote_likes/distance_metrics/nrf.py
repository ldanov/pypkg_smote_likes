#!/usr/bin/env python3

"""
    Nearest reference features
"""

# Authors: Lyubomir Danov <->
# License: -


from sklearn.utils import safe_indexing
import numpy
from scipy import stats


class NearestReferenceFeatures(object):
    def __init__(self):
        pass

    def closest_nonclass_values(self, X_interest, X_reference):
        '''
            Return closest bigger and smaller values from X_reference 
            for each element in X_interest

            If a bigger/smaller value is not found, the original value 
            is returned.
        '''

        X_bigger = self._closest_nonclass(X_interest, X_reference, 'bigger')
        X_smaller = self._closest_nonclass(X_interest, X_reference, 'smaller')

        return {'bigger': X_bigger, 'smaller': X_smaller}

    def _closest_nonclass(self, X_interest, X_reference, comparison_type):

        if comparison_type == 'bigger':
            x = 1
        elif comparison_type == 'smaller':
            x = -1

        X_close = numpy.zeros(X_interest.shape, dtype=X_interest.dtype)
        single_sample_shape = (1, X_interest.shape[1])

        for class_row in range(X_interest.shape[0]):
            X_temp = numpy.vstack([X_interest[class_row, :], X_reference])

            X_tmp_rank = numpy.apply_along_axis(stats.rankdata, axis=0,
                                                arr=X_temp, method='dense')

            tmp_features = numpy.zeros(single_sample_shape,
                                       dtype=X_interest.dtype)

            for col, rank in enumerate(X_tmp_rank[0, :]):
                row_loc = numpy.where(X_tmp_rank[:, col] == (rank + x))[0]
                if row_loc.shape == (0,):
                    row_loc = numpy.where(X_tmp_rank[:, col] == rank)[0]
                row_id = row_loc[0]
                tmp_features[0, col] = X_temp[row_id, col]

            X_close[class_row, :] = tmp_features

        return X_close
