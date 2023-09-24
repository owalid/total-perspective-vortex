import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class TurboCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.W_ = None
        self.mean_ = 0
        self.std_ = 0
    
    def _get_covariance_matrix(self, X, y):
        unique_classes = np.unique(y)

        class_covariances = []
        for unique_class in unique_classes:
            class_data = X[y == unique_class]
            _, n_channels, _ = class_data.shape

            '''
                We need to transpose the data to (n_channels, n_epochs, n_times),
                because the covariance matrix function in numpy takes the last dimension as the variable dimension
                and the other dimensions as the observation dimension.
                https://numpy.org/doc/stable/reference/generated/numpy.cov.html
            '''
            class_data = np.transpose(class_data, [1,0,2]) # Transpose to (n_channels, n_epochs, n_times)
            class_data = class_data.reshape(n_channels, -1) # Reshape to (n_channels, n_epochs * n_times)
            class_data = np.cov(class_data) # covariance matrix
            class_covariances.append(class_data) # append to list

        res = np.stack(class_covariances)
        return res
    
    def fit(self, X, y):
        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two distinct values.")
        
        covs = self._get_covariance_matrix(X, y) # get covariance matrix

        # compute eigenvalues and eigenvectors
        eigen_values, eigen_vectors = eigh(covs[0], covs.sum(axis=0))
        sort_idx = np.argsort(np.abs(eigen_values - 0.5))[::-1] # sort eigenvalues with respect to the distance from 0.5 (0.5 is the mean of the eigenvalues)

        # Get eigenvalues and eigenvectors sorted by the distance from 0.5
        # The first components are supposed to contain the most discriminative information between the classes.
        eigen_values = eigen_values[sort_idx]
        eigen_vectors = eigen_vectors[:, sort_idx]
        
        W = eigen_vectors.T # transpose to (n_components, n_channels)

        # get filters
        self.W_ = W[:self.n_components] # get first n_components filters
        X = np.asarray([np.dot(self.W_, epoch) for epoch in X]) # apply filters to data

        # compute features (mean power)
        X = (X**2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    
    def transform(self, X):
        X = np.asarray([np.dot(self.W_, epoch) for epoch in X])
        X = (X**2).mean(axis=2)
        X -= self.mean_
        X /= self.std_

        return X
