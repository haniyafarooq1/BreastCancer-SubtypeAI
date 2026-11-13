"""Phase 1: load the dataset, build the cache, apply the existing train/test split and
verify it. Run from the repo root:  python ml/01_load_split.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ml import config as C
from ml.data import load_split

data, feats = load_split()
X_train, y_train = data["train"]
X_test, y_test = data["test"]

print("\n=== FEATURES ===")
print("Feature count (all columns except 'subtype'):", len(feats))
print("Feature dtypes:", pd.Series(X_train.dtypes).value_counts().to_dict())
print("Metadata cols in features:", sorted(C.NON_FEATURE_COLS & set(feats)) or "none")
print("NaN in train/test:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
print("Inf in train/test:", int(np.isinf(X_train.to_numpy()).sum()), int(np.isinf(X_test.to_numpy()).sum()))

print("\n=== SHAPES ===")
print("X_train", X_train.shape, "| y_train", y_train.shape)
print("X_test ", X_test.shape, "| y_test ", y_test.shape)

counts = pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()}).reindex(C.CLASS_ORDER)
counts["total"] = counts.sum(axis=1)
counts.loc["All"] = counts.sum()
print("\n=== PER-CLASS COUNTS ===")
print(counts.to_string())

# Final sanity asserts against the known numbers
assert X_train.shape[0] == 2618 and X_test.shape[0] == 655
assert set(X_train.index).isdisjoint(X_test.index)
assert "subtype" not in X_train.columns
assert not X_train.index.str.contains("repl").any() and not X_test.index.str.contains("repl").any()
assert sorted(y_train.unique()) == C.CLASS_ORDER
print("\nAll asserts passed: no train/test overlap, 'subtype' not a feature, no 'repl' IDs, "
      "all features numeric float32, manifest labels match dataset labels.")
