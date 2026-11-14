"""Pipeline factory: TopVariance(5000) -> StandardScaler -> SelectKBest(f_classif) -> model.

All preprocessing lives inside the Pipeline, so GridSearchCV re-fits it on every fold's
training part (no leakage). `memory` caches fitted transformers across hyper-parameter
combinations of the final estimator.
"""
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import PIPELINE_CACHE  # resolved portably in ml/config.py
from ml.transformers import TopVarianceSelector


def build_pipeline(model, n_top=5000, k=1000, memory=PIPELINE_CACHE):
    return Pipeline(
        steps=[
            ("variance", TopVarianceSelector(n_top=n_top)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("model", model),
        ],
        memory=str(memory) if memory is not None else None,
    )
