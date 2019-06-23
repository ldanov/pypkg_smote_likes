#!/usr/bin/env python3

"""Prototype of restricted-SMOTE v1"""

# Authors: Lyubomir Danov <->
# License: -


import numpy
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_sampling_strategy
from scipy import sparse
from sklearn.utils import check_random_state, safe_indexing

from ..distance_metrics.nrf import NearestReferenceFeatures


class restrictedSMOTE1(BaseOverSampler):
    """Class to perform over-sampling using SMOTE.

    Parameters
    ----------
    sampling_strategy: dict 
        The key gives the class to be over-sampled, while the value
        gives how many synthetic observations to be generated.

    beta_bigger_params : dict with keys 'a' and 'b'
        Which parameters to use when sampling from numpy.random.beta
        for bigger samples. For smaller samples the parameters are reversed.

    bigger_samples: dict or None
        The key gives the class to be over-sampled, while the value
        gives how many synthetic observations will be generated bigger.
        If None, the numbers will be randomly sampled.

    random_state : int, RandomState instance or None, optional (default=None)
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.



    """
    _required_beta_keys = ['a', 'b']

    def __init__(self,
                 sampling_strategy: dict = {'class': 'number'},
                 beta_bigger_params={'a': 5, 'b': 2},
                 bigger_samples=None,
                 random_state=None,
                 n_jobs=1):
        if not isinstance(sampling_strategy, dict):
            raise TypeError('Only dict is supported for sampling_strategy')

        super().__init__(sampling_strategy=sampling_strategy,
                         ratio=None)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._sampling_type = 'over-sampling'
        self.nrf_ = NearestReferenceFeatures()
        if bigger_samples is None:
            bigger_samples = {}
        self.bigger_samples = bigger_samples

        self._validate_beta_params(beta_bigger_params)

    def _fit_resample(self, X, y):
        self._validate_estimator()
        return self._sample(X, y)

    def _validate_estimator(self):
        pass

    def _sample(self, X, y):

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            # return the nearest
            # (positive and negative) non-preference class
            # features as 2 ndarrays
            target_class_indices = numpy.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            target_nonclass_indices = numpy.flatnonzero(y != class_sample)
            X_nonclass = safe_indexing(X, target_nonclass_indices)

            X_bigger, X_smaller = self.nrf_.closest_nonclass_values(X_interest=X_class,
                                                                    X_reference=X_nonclass)

            n_big, n_small = self._n_samples(self.bigger_samples.get(class_sample, None),
                                             n_samples)

            nn_num = numpy.arange(X_class.shape[0])
            nn_num = nn_num.reshape(X_class.shape[0], 1)

            X_new_bigger = self._make_samples(X_class, X_bigger, nn_num,
                                              n_big, 1.0, 'bigger')
            X_new_smaller = self._make_samples(X_class, X_smaller, nn_num,
                                               n_small, 1.0, 'smaller')

            X_new = numpy.vstack([X_new_bigger, X_new_smaller])
            y_new = numpy.array([class_sample] * n_samples, dtype=y.dtype)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = numpy.vstack((X_resampled, X_new))
            y_resampled = numpy.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def _make_samples(self,
                      X,
                      nn_data,
                      nn_num,
                      n_samples,
                      step_size=1.,
                      beta_sampling='bigger'):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.



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
        beta_a = self.beta_params['a_' + beta_sampling]
        beta_b = self.beta_params['b_' + beta_sampling]
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = step_size * random_state.beta(a=beta_a,
                                              b=beta_b,
                                              size=n_samples)
        rows = numpy.floor_divide(samples_indices, nn_num.shape[1])
        cols = numpy.mod(samples_indices, nn_num.shape[1])

        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                if X[row].nnz:
                    sample = self._generate_sample(X, nn_data, nn_num,
                                                   row, col, step)
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()
            return sparse.csr_matrix((samples, (row_indices, col_indices)),
                                     [len(samples_indices), X.shape[1]],
                                     dtype=X.dtype)
        else:
            X_new = numpy.zeros((n_samples, X.shape[1]), dtype=X.dtype)
            X_new = self._generate_sample(X, nn_data, nn_num,
                                          rows, cols, steps[:, None])
            return X_new

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

    def _n_samples(self, bigger_samples, n_samples):
        if bigger_samples is None:
            # randomly select how to distribute n_samples between
            # bigger and smaller
            random_state = check_random_state(self.random_state)
            n_big = random_state.randint(low=0, high=n_samples+1, size=1)[0]
            n_small = n_samples - n_big
        else:
            n_big = bigger_samples
            n_small = n_samples - n_big

        return n_big, n_small

    def _validate_beta_params(self, beta_bigger_params):
        if not isinstance(beta_bigger_params, dict):
            raise TypeError(
                'Only explicit dict is supported for beta_bigger_params')

        missing_keys = [
            bkey for bkey in self._required_beta_keys if bkey not in beta_bigger_params.keys()]

        if len(missing_keys) != 0:
            raise KeyError(
                'beta_bigger_params does not have following keys:{}'.format(missing_keys))

        beta_smaller_params = self._invert_beta_params(beta_bigger_params)
        self.beta_params = {
            'a_bigger': beta_bigger_params['a'],
            'b_bigger': beta_bigger_params['b'],
            'a_smaller': beta_smaller_params['a'],
            'b_smaller': beta_smaller_params['b']
        }

    def _invert_beta_params(self, beta_dict):
        return {'a': beta_dict['b'], 'b': beta_dict['a']}
