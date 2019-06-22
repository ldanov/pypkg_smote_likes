import numpy
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.utils import check_neighbors_object, check_sampling_strategy, check_target_type
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, safe_indexing, check_X_y, check_array
from imblearn.pipeline import Pipeline
from scipy import sparse
from sklearn.utils.sparsefuncs_fast import csc_mean_variance_axis0
from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
from sklearn.preprocessing import OneHotEncoder
from collections import Counter




from ..distance_metrics import hvdm, normalized_vdm_2







class MySMOTE(SMOTE):
    """

    Parameters
    ----------
    categorical_features : ndarray, shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

   

    """

    def __init__(self, categorical_features, sampling_strategy='auto',
                 random_state=None, k_neighbors=5, n_jobs=1):
        super(MySMOTE, self).__init__(sampling_strategy=sampling_strategy,
                                      random_state=random_state,
                                      k_neighbors=k_neighbors,
                                      ratio=None)
        self.categorical_features = categorical_features
        self.kind = 'regular'

    def _set_neighbours_object(self, additional_neighbor=0):
        self.nn_k_ = NearestNeighbors(n_neighbors = (self.k_neighbors + \
            additional_neighbor), metric='precomputed', algorithm='brute')
        self.nn_k_.set_params(**{'n_jobs': self.n_jobs})

    # overwrite SMOTE's _sample
    def _my_sample(self, X, y):
        # FIXME: uncomment in version 0.6
        # self._validate_estimator()

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

    def _fit_resample(self, X, y):
        self.n_features_ = X.shape[1]
        self._validate_estimator()

        # compute the median of the standard deviation of the minority class
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_minority = safe_indexing(X_continuous,
                                   numpy.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == 'csr':
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = numpy.median(numpy.sqrt(var))

        X_categorical = X[:, self.categorical_features_]
        if X_continuous.dtype.name != 'object':
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = numpy.float64
        self.ohe_ = OneHotEncoder(sparse=True, handle_unknown='ignore',
                                  dtype=dtype_ohe)
        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray() if sparse.issparse(X_categorical)
            else X_categorical)

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.
        X_ohe.data = (numpy.ones_like(X_ohe.data, dtype=X_ohe.dtype) *
                      self.median_std_ / 2)
        X_encoded = sparse.hstack((X_continuous, X_ohe), format='csr')

        # use SMOTE._fit_resample
        X_resampled, y_resampled = self._my_sample(X_encoded, y)

        # reverse the encoding of the categorical features
        X_res_cat = X_resampled[:, self.continuous_features_.size:]
        X_res_cat.data = numpy.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat)

        if sparse.issparse(X):
            X_resampled = sparse.hstack(
                (X_resampled[:, :self.continuous_features_.size],
                 X_res_cat_dec), format='csr'
            )
        else:
            X_resampled = numpy.hstack(
                (X_resampled[:, :self.continuous_features_.size].toarray(),
                 X_res_cat_dec)
            )

        indices_reordered = numpy.argsort(
            numpy.hstack((self.continuous_features_, self.categorical_features_))
        )
        if sparse.issparse(X_resampled):
            # the matrix is supposed to be in the CSR format after the stacking
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]

        return X_resampled, y_resampled

    def _generate_sample_base(self, X, nn_data, nn_num, row, col, step):
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

    @staticmethod
    def _check_X_y(X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=None)
        return X, y, binarize_y

    def _validate_estimator(self):
        self._set_neighbours_object(additional_neighbor=1)
        categorical_features = numpy.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            self.categorical_features_ = numpy.flatnonzero(categorical_features)
        else:
            if any([cat not in numpy.arange(self.n_features_)
                    for cat in categorical_features]):
                raise ValueError(
                    'Some of the categorical indices are out of range. Indices'
                    ' should be between 0 and {}'.format(self.n_features_))
            self.categorical_features_ = categorical_features
        self.continuous_features_ = numpy.setdiff1d(numpy.arange(self.n_features_),
                                                 self.categorical_features_)

    

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        """Generate a synthetic sample with an additional steps for the
        categorical features.

        Each new sample is generated the same way than in SMOTE. However, the
        categorical features are mapped to the most frequent nearest neighbors
        of the majority class.
        """
        rng = check_random_state(self.random_state)
        sample = self._generate_sample_base(X, nn_data, nn_num,
                                                       row, col, step)
        # To avoid conversion and since there is only few samples used, we
        # convert those samples to dense array.
        return self._format_sample(sample, nn_data, nn_num, row, rng, X)


    def _format_sample(self, sample, nn_data, nn_num, row, rng, X):
        # To avoid conversion and since there is only few samples used, we
        # convert those samples to dense array.
        sample = (sample.toarray().squeeze()
                  if sparse.issparse(sample) else sample)
        all_neighbors = nn_data[nn_num[row]]
        all_neighbors = (all_neighbors.toarray()
                         if sparse.issparse(all_neighbors) else all_neighbors)

        categories_size = ([self.continuous_features_.size] +
                           [cat.size for cat in self.ohe_.categories_])

        for start_idx, end_idx in zip(numpy.cumsum(categories_size)[:-1],
                                      numpy.cumsum(categories_size)[1:]):
            col_max = all_neighbors[:, start_idx:end_idx].sum(axis=0)
            # tie breaking argmax
            col_sel = rng.choice(numpy.flatnonzero(
                numpy.isclose(col_max, col_max.max())))
            sample[start_idx:end_idx] = 0
            sample[start_idx + col_sel] = 1

        return sparse.csr_matrix(sample) if sparse.issparse(X) else sample

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
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                X_new[i] = self._generate_sample(X, nn_data, nn_num,
                                                 row, col, step)
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