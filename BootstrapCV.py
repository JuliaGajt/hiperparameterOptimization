import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import resample


class BootstrapCV(BaseCrossValidator):
    def __init__(self, n_bootstrap_samples):
        self.n_bootstrap_samples = n_bootstrap_samples

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        for _ in range(self.n_bootstrap_samples):
            # Losowe próbkowanie z powtórzeniami za pomocą metody resample
            bootstrap_indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
            train_indices = bootstrap_indices
            test_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_bootstrap_samples
