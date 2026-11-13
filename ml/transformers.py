"""Custom sklearn transformers."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class TopVarianceSelector(TransformerMixin, BaseEstimator):
    """Keep the `n_top` columns with the highest variance, computed in `fit` only.

    Because it lives inside the Pipeline, it is re-fit on each CV fold's training part,
    so the test fold / test set never influences which genes are kept.
    Output is float32 to halve memory for downstream steps.
    """

    def __init__(self, n_top=5000):
        self.n_top = n_top

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float32, np.float64])
        n_top = min(self.n_top, X.shape[1])
        self.variances_ = X.var(axis=0, dtype=np.float64)
        # stable sort => deterministic tie-breaking; keep original column order in output
        order = np.argsort(-self.variances_, kind="stable")[:n_top]
        self.support_idx_ = np.sort(order)
        return self

    def transform(self, X):
        check_is_fitted(self, "support_idx_")
        X = validate_data(self, X, dtype=[np.float32, np.float64], reset=False)
        return X[:, self.support_idx_].astype(np.float32, copy=False)

    def get_support(self, indices=False):
        check_is_fitted(self, "support_idx_")
        if indices:
            return self.support_idx_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_idx_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_idx_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            input_features = np.array([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features, dtype=object)[self.support_idx_]
