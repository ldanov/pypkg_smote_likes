#!/usr/bin/env python3

"""Prototype rebuilding of imblearn.over_sampling.SMOTE"""

# Authors: Lyubomir Danov <->
# License: -


import numpy
from imblearn.over_sampling._smote import BaseSMOTE
from imblearn.utils import check_neighbors_object
from scipy import sparse
from sklearn.utils import check_random_state, safe_indexing


class protoSMOTE(BaseSMOTE):
    """Class to perform over-sampling using SMOTE.

    Parameters
    ----------
    sampling_strategy: dict
        The key gives the class to be over-sampled, while the value
        gives how many synthetic observations to be generated.

    random_state : int, RandomState instance or None, optional (default=None)
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    k_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    """

    def __init__(self,
                 sampling_strategy: dict = {'class': 'number'},
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=1):
        if not isinstance(sampling_strategy, dict):
            raise TypeError('Only explicit dict is supported')
        super().__init__(sampling_strategy=sampling_strategy,
                         random_state=random_state, k_neighbors=k_neighbors,
                         n_jobs=n_jobs, ratio=None)
        self.n_jobs = n_jobs

    def _fit_resample(self, X, y):

        self._validate_estimator()
        return self._sample(X, y)

    def _sample(self, X, y):

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = numpy.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(X_class, y.dtype, class_sample,
                                              X_class, nns, n_samples, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = numpy.vstack((X_resampled, X_new))
            y_resampled = numpy.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def _make_samples(self,
                      X,
                      y_dtype,
                      y_type,
                      nn_data,
                      nn_num,
                      n_samples,
                      step_size=1.):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, optional (default=1.)
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.

        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = step_size * random_state.uniform(size=n_samples)
        rows = numpy.floor_divide(samples_indices, nn_num.shape[1])
        cols = numpy.mod(samples_indices, nn_num.shape[1])

        y_new = numpy.array([y_type] * len(samples_indices), dtype=y_dtype)

        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                if X[row].nnz:
                    sample = self._generate_sample(X, nn_data, nn_num,
                                                   row, col, step)
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()
            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [len(samples_indices), X.shape[1]],
                                      dtype=X.dtype),
                    y_new)
        else:
            X_new = numpy.zeros((n_samples, X.shape[1]), dtype=X.dtype)
            X_new = self._generate_sample(X, nn_data, nn_num,
                                          rows, cols, steps[:, None])
            return X_new, y_new

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        row : int
            Index pointing at feature vector in X which will be used
            as a base for creating new sample.

        col : int
            Index pointing at which nearest neighbor of base feature vector
            will be used when creating new sample.

        step : float
            Step size for new sample.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_features,)
            Single synthetically generated sample.

        """
        return X[row] - step * (X[row] - nn_data[nn_num[row, col]])
