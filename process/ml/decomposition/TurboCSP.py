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
            class_data = np.transpose(class_data, [1,0,2])
            class_data = class_data.reshape(n_channels, -1)
            class_data = np.cov(class_data)
            class_covariances.append(class_data)

        return np.stack(class_covariances)
    
    def fit(self, X, y):
        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two distinct values.")
        covs = self._get_covariance_matrix(X, y)

        eigen_values, eigen_vectors = eigh(covs[0], covs.sum(axis=0))
        sort_idx = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        eigen_values = eigen_values[sort_idx]
        eigen_vectors = eigen_vectors[:, sort_idx]
        
        W = eigen_vectors.T
    
        self.W_ = W[:self.n_components]
        X = np.asarray([np.dot(self.W_, epoch) for epoch in X])

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
