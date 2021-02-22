"""Custom principle component analysis module.

The code was modified by Zhonghua Zheng based on 
https://github.com/alfredsasko/advanced-principle-component-analysis/issues/3

Iplemented customizations
- varimax rotation for better interpretation of principal components
- communalities calculation for selecting significant features
- loading significant threshold comparison as function of sample size
  for selecting significant features
- 'surrogate' feature selection used for dimensionality reduction -
  features with maximum laoding instead of principal components are selected
"""

# IMPORTS
# -------

# Standard libraries
import numbers

# 3rd party libraries
import numpy as np
from numpy import linalg
import pandas as pd
from scipy.sparse import issparse

import rpy2
import rpy2.rlike.container as rlc
from rpy2 import robjects
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.vectors import ListVector
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

from sklearn.decomposition import PCA
from sklearn.utils import check_array
from sklearn.utils.extmath import svd_flip
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted

class CustomPCA(PCA):
    '''Customized PCA with following options
       - varimax rotation
       - different feature selection methods

       and calculated communalities.
    '''

    def __init__(self, rotation=None, feature_selection='all',  **kws):
        '''
        rotation: string, name of the rotation method.
                  'varimax': varimax rotation
                  None: no rotation by by default

        feature_selection: string, Features selection method
                           'all': All features are selected, means principal
                                  components are output of tranformation
                           'significant': Only features with significant weights
                                          and communalities are selected
                           'surrogate': Only features with highest loading
                                        with principal components are used
                           'summated scales': Summated scales consturcted as
                                              sum of features having significant
                                              loadings on principal components,
                                              not implemented yet.
        '''

        super().__init__(**kws)
        self.rotation = rotation
        self.feature_selection = feature_selection

    @staticmethod
    def _df2mtr(df):
        '''Convert pandas dataframe to r matrix. Category dtype is casted as
        factorVector considering missing values
        (original py2ri function of rpy2 can't handle this properly so far)

        Args:
            data: pandas dataframe of shape (# samples, # features)
                  with numeric dtype

        Returns:
            mtr: r matrix of shape (# samples # features)
        '''
        # check arguments
        assert isinstance(df, pd.DataFrame), 'Argument df need to be a pd.Dataframe.'

        # select only numeric columns
        df = df.select_dtypes('number')

        # create and return r matrix
        values = FloatVector(df.values.flatten())
        dimnames = ListVector(
            rlc.OrdDict([('index', StrVector(tuple(str(x) for x in df.index))),
            ('columns', StrVector(tuple(str(x) for x in df.columns)))])
        )

        return robjects.r.matrix(values, nrow=len(df.index), ncol=len(df.columns),
                                 dimnames = dimnames, byrow=True)

    def _varimax(self, factor_df, **kwargs):
        '''
        varimax rotation of factor matrix

        Args:
            factor_df: factor matrix as pd.DataFrame with shape
                       (# features, # principal components)

        Return:
            rot_factor_df: rotated factor matrix as pd.DataFrame
        '''
        factor_mtr = self._df2mtr(factor_df)
        varimax = robjects.r['varimax']
        rot_factor_mtr = varimax(factor_mtr)
        return pandas2ri.rpy2py(rot_factor_mtr.rx2('loadings'))

    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""
        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy)

        # Handle n_components==None
        if self.n_components is None:
            if self.svd_solver != 'arpack':
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == 'auto':
            # Small problem or n_components == 'mle', just call full PCA
            if (max(X.shape) <= 500
                or n_components == 'mle'
                or n_components == 'latent_root'):
                self._fit_svd_solver = 'full'
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                self._fit_svd_solver = 'randomized'
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = 'full'

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == 'full':
            U, S , V = self._fit_full(X, n_components)
        elif self._fit_svd_solver in ['arpack', 'randomized']:
            U, S, V = self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'"
                             "".format(self._fit_svd_solver))

        # implmentation of varimax rotation
        if self.rotation == 'varimax':
            if self.n_samples_ > self.n_components_:

                factor_matrix = (
                    self.components_.T
                    * (self.explained_variance_.reshape(1, -1) ** (1/2))
                )

                rot_factor_matrix = self._varimax(pd.DataFrame(factor_matrix))

                self.explained_variance_ = (rot_factor_matrix ** 2).sum(axis=0)

                self.components_ = (
                    rot_factor_matrix
                    / (self.explained_variance_.reshape(1, -1) ** (1/2))
                ).T

                # sort components by explained variance in descanding order
                self.components_ = self.components_[
                    np.argsort(self.explained_variance_)[::-1], :
                ]

                self.explained_variance_ = np.sort(
                    self.explained_variance_
                )[::-1]

                total_var = self.n_features_
                self.explained_variance_ratio_ = (
                        self.explained_variance_ / total_var
                )

                self.singular_values_ = None

                if self._fit_svd_solver == 'full':
                    if self.n_components_ < min(self.n_features_, self.n_samples_):
                        self.noise_variance_ = (
                            (total_var - self.explained_variance_.sum())
                            / (self.n_features_ - self.n_components_)
                        )
                    else:
                        self.noise_variance_ = 0.

                elif self._fit_svd_solver in ['arpack', 'randomized']:
                    if self.n_components_ < min(self.n_features_, self.n_samples_):

                        total_var = np.var(X, ddof=1, axis=0)

                        self.noise_variance_ = (
                            total_var.sum() - self.explained_variance_.sum()
                        )

                        self.noise_variance_ /= (
                            min(self.n_features_, self.n_samples_)
                            - self.n_components_
                        )
                    else:
                        self.noise_variance_ = 0.

                else:
                    raise ValueError("Unrecognized svd_solver='{0}'"
                                     "".format(self._fit_svd_solver))
            else:
                raise ValueError(
                    "Varimax rotation requires n_samples > n_components")

            U, S, V = None, None, None

        # implmentation of communalties
        self.communalities_ = (self.components_ ** 2).sum(axis=0)

        return U, S, V

    def _fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X"""
        n_samples, n_features = X.shape

        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")
        elif n_components == 'latent_root':
            pass
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='full'"
                             % (n_components, min(n_samples, n_features)))
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError("n_components=%r must be of type int "
                                 "when greater than or equal to 1, "
                                 "was of type=%r"
                                 % (n_components, type(n_components)))

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        components_ = V

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        # Postprocess the number of components required
        if n_components == 'mle':
            n_components = \
                _infer_dimension_(explained_variance_, n_samples, n_features)
        elif n_components == 'latent_root':
            n_components = (explained_variance_ > 1).sum()
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()

        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, V

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected
        Parameters
        ----------
        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.
        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

        attrs = [v for v in vars(self)
                 if (v.endswith("_") or v.startswith("_"))
                 and not v.startswith("__")]
        check_is_fitted(self, attributes=attrs,
                        all_or_any=all)

        # Keep only features with at least one of the significant loading
        # and communality > 0.5
        significant_features_mask = (
            ((np.absolute(self.components_)
              >= self.calculate_significance_threshold())
             .any(axis=0))
            & (self.communalities_ >= 0.5)
        )

        if self.feature_selection == 'all':
            mask = np.array([True] * self.n_features_)

        elif self.feature_selection == 'significant':
            mask = significant_features_mask

        elif self.feature_selection == 'surrogate':
            # Select significant feature with maximum loading
            # on each principal component
            mask = np.full(self.n_features_, False, dtype=bool)
            surrogate_features_idx = np.unique(
                np.absolute(np.argmax(self.components_, axis=1))
            )
            mask[surrogate_features_idx] = True
            mask = (mask & significant_features_mask)

        elif self.feature_selection == 'summated scales':
            raise Exception('Not implemented yet.')

        else:
            raise ValueError('Not valid selection method.')
        return mask


    def calculate_significance_threshold(self):
        sample_sizes = np.array([50, 60, 70, 85, 100,
                                 120, 150, 200, 250, 300])
        thresholds = np.array([0.75, 0.70, 0.65, 0.60, 0.55,
                               0.50, 0.45, 0.40, 0.35, 0.30])
        return min(thresholds[sample_sizes <= self.n_samples_])

    def transform(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, n_components=2)
        >>> ipca.transform(X) # doctest: +SKIP
        """
        attrs = [v for v in vars(self)
                 if (v.endswith("_") or v.startswith("_"))
                 and not v.startswith("__")]
        check_is_fitted(self, attributes=attrs,
                        all_or_any=all)

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_

        if self.feature_selection == 'all':
            X_transformed = np.dot(X, self.components_.T)
            if self.whiten:
                X_transformed /= np.sqrt(self.explained_variance_)

        else:
            X_transformed = X[:, self._get_support_mask()]

        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Ignored variable.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """

        U, S, V = self._fit(X)
        X_transformed = self.transform(X)

        return X_transformed

    def count_cross_loadings(self):
        '''Calculates number of cross loadings'''
        is_significant = (np.absolute(self.components_)
                          > self.calculate_significance_threshold())
        count = is_significant.sum()
        return count
