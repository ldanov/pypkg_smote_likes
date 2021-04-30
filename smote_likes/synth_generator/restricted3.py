#!/usr/bin/env python3

"""Prototype of restricted-SMOTE v3"""

# Authors: Lyubomir Danov <->
# License: -


import numpy
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_sampling_strategy
from sklearn.utils import check_random_state

from ..distance_metrics.nrf import NearestReferenceFeatures
from .utils import safe_indexing


class restrictedSMOTE3(BaseOverSampler):
    """Class to perform over-sampling using SMOTE.

    Parameters
    ----------
    sampling_strategy : dict
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
    # TODO: expose methods in documentation
    _required_beta_keys = ['a', 'b']

    def __init__(self,
                 sampling_strategy: dict = {'class': 'number'},
                 beta_params={'a': 1, 'b': 1},
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

        self._validate_beta_params(beta_params)

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

            X_new = self._make_samples(X_class, X_bigger, X_smaller,
                                       n_samples, 1.0)

            y_new = numpy.array([class_sample] * n_samples, dtype=y.dtype)

            X_resampled = numpy.vstack((X_resampled, X_new))
            y_resampled = numpy.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def _make_samples(self,
                      X,
                      X_big,
                      X_small,
                      n_samples,
                      step_size=1.):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        X_big : {array-like}, shape (n_samples, n_features)
            Points from which the points will be created.

        X_small : {array-like}, shape (n_samples, n_features)
            Points from which the points will be created.

        n_samples : int
            The number of samples to generate.

        step_size : float, optional (default=1.)
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray}, shape (n_samples_new, n_features)
            Synthetically generated samples.

        """
        beta_a = self.beta_params['a']
        beta_b = self.beta_params['b']
        random_state = check_random_state(self.random_state)
        rows = random_state.randint(low=0,
                                    high=X.shape[0],
                                    size=n_samples)
        steps = step_size * random_state.beta(a=beta_a,
                                              b=beta_b,
                                              size=n_samples)

        X_new = numpy.zeros((n_samples, X.shape[1]), dtype=X.dtype)
        X_new = self._generate_sample(X, X_big, X_small,
                                      rows, steps[:, None])
        return X_new

    def _generate_sample(self, X, X_big, X_small, row, step):
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
        X : {array-like}, shape (n_samples, n_features)
            Points from which the points will be created.

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
        X_new : {ndarray}, shape (n_features,)
            Single synthetically generated sample.

        """
        
        len_down = numpy.subtract(X, X_small)[row, :]
        len_up = numpy.subtract(X_big, X)[row, :]
        len_eq = numpy.zeros_like(len_down)

        len_fin = numpy.where(step > 0.5, len_up, len_eq)
        len_fin = numpy.where(step < 0.5, len_down, len_fin)

        return X[row, :] + step * len_fin

    def _validate_beta_params(self, beta_params):
        if not isinstance(beta_params, dict):
            raise TypeError(
                'Only explicit dict is supported for beta_params')

        missing_keys = [
            bkey for bkey in self._required_beta_keys if bkey not in beta_params.keys()]

        if len(missing_keys) != 0:
            raise KeyError(
                'beta_params does not have following keys:{}'.format(missing_keys))

        # a/b constraint: a == b
        if beta_params['a']!=beta_params['b']:
            raise ValueError('beta_params a and b must be equal')

        self.beta_params = {
            'a': beta_params['a'],
            'b': beta_params['b']
        }
        """[summary]
        """
