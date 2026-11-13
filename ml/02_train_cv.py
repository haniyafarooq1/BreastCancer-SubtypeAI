"""Phase 3: tune LR / SVC / RF with 5-fold GridSearchCV on the TRAINING set only,
save CV results, per-class CV F1, and the final pipelines (fit on full training set).

The test set is never loaded here.
Run from repo root:  python ml/02_train_cv.py
Log: results/ml/cv/train_log.txt
"""
import json
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from ml import config as C
from ml.data import load_split
from ml.pipeline import PIPELINE_CACHE, build_pipeline

# BC_DRY_RUN=<dir>: quick bug-check on a 400-sample subsample, outputs go to <dir>
import os
DRY = os.environ.get("BC_DRY_RUN")
if DRY:
    C.CV_DIR = Path(DRY) / "cv"
    C.MODELS_DIR = Path(DRY) / "models"

C.CV_DIR.mkdir(parents=True, exist_ok=True)
C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = C.CV_DIR / "train_log.txt"


class TimestampTee:
    """Duplicate a stream to the log file, prefixing each line with a timestamp."""
    def __init__(self, stream, fh):
        self.stream, self.fh, self.at_line_start = stream, fh, True

    def write(self, s):
        self.stream.write(s)
        out = []
        for part in s.splitlines(keepends=True):
            if self.at_line_start and part.strip():
                out.append(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))
            out.append(part)
            self.at_line_start = part.endswith("\n")
        self.fh.write("".join(out)); self.fh.flush()

    def flush(self):
        self.stream.flush(); self.fh.flush()


log_fh = open(LOG_PATH, "a", encoding="utf-8")
sys.stdout = TimestampTee(sys.__stdout__, log_fh)
sys.stderr = TimestampTee(sys.__stderr__, log_fh)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)  # scipy L-BFGS-B 'disp' noise
warnings.simplefilter("default", ConvergenceWarning)           # but keep convergence issues in the log

T_START = time.time()
print(f"\n{'#' * 70}\nPHASE 3 TRAINING RUN  (sklearn {sklearn.__version__})\n{'#' * 70}")

# ------------------------------------------------------------------ data
data, feats = load_split(which=("train",))
X, y = data["train"]
if DRY:
    X = X.groupby(y, group_keys=False).sample(80, random_state=0)
    y = y.loc[X.index]
print(f"X_train {X.shape}, classes {sorted(y.unique())}; test set NOT loaded")

CV = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
SCORING = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
K_GRID = [500, 1000] if DRY else [500, 1000, 2000]
shutil.rmtree(PIPELINE_CACHE, ignore_errors=True)

# ------------------------------------------------------------------ model specs
SPECS = {
    "logreg": dict(
        estimator=LogisticRegression(class_weight="balanced", max_iter=3000,
                                     random_state=C.RANDOM_STATE),
        grid=[{"select__k": K_GRID, "model__C": [0.001, 0.01, 0.1, 1]}],
    ),
    "svc": dict(
        estimator=SVC(class_weight="balanced", random_state=C.RANDOM_STATE),  # no probability in CV
        grid=[
            {"select__k": K_GRID, "model__kernel": ["linear"], "model__C": [0.001, 0.01, 0.1]},
            {"select__k": K_GRID, "model__kernel": ["rbf"], "model__C": [1, 10], "model__gamma": ["scale"]},
        ],
    ),
    "rf": dict(
        estimator=RandomForestClassifier(n_estimators=300, max_features="sqrt", max_depth=None,
                                         class_weight="balanced", n_jobs=2,  # threads, no data copy
                                         random_state=C.RANDOM_STATE),
        grid=[{"select__k": K_GRID, "model__min_samples_leaf": [1, 3]}],
    ),
}


def grid_edge_warnings(name, best, grid):
    """Flag numeric hyper-parameters (C, k) whose best value is at the edge of its grid."""
    warns = []
    # use the sub-grid that contains the best params (relevant for SVC linear vs rbf)
    sub = next(g for g in grid if all(best.get(p) in v for p, v in g.items()))
    for p in ("model__C", "select__k"):
        if p not in sub or len(sub[p]) < 2:
            continue
        vals, b = sorted(sub[p]), best[p]
        if b == vals[0] or b == vals[-1]:
            side = "lower" if b == vals[0] else "upper"
            if p == "model__C":
                ext = b / 10 if side == "lower" else b * 10
            else:
                ext = 250 if side == "lower" else 3000
            warns.append(f"{name}: best {p}={b} is at the {side} edge of {vals}; "
                         f"suggested extension: add {p}={ext:g}")
    return warns


def final_estimator(name, best_pipe):
    """Final pipeline refit on full training data. SVC gets sigmoid calibration so that
    predict == argmax(predict_proba)."""
    pipe = clone(best_pipe)
    pipe.set_params(memory=None)
    if name == "svc":
        svc = pipe.named_steps["model"]
        pipe.set_params(model=CalibratedClassifierCV(svc, cv=5, method="sigmoid"))
    return pipe


summary_rows, per_class_rows, all_warnings = [], [], []

for name, spec in SPECS.items():
    print(f"\n{'=' * 70}\nMODEL: {name}\n{'=' * 70}")
    n_combos = sum(int(np.prod([len(v) for v in g.values()])) for g in spec["grid"])
    print(f"{n_combos} combos x {C.CV_FOLDS} folds = {n_combos * C.CV_FOLDS} fits (+1 refit)")
    gs = GridSearchCV(
        build_pipeline(spec["estimator"]), spec["grid"], scoring=SCORING, refit="f1_macro",
        cv=CV, n_jobs=1, pre_dispatch="n_jobs", verbose=3, error_score="raise",
        return_train_score=False,
    )
    t0 = time.time(); gs.fit(X, y); t_search = time.time() - t0
    print(f"{name}: search done in {t_search / 60:.1f} min | best {gs.best_params_} "
          f"| macro F1 {gs.best_score_:.4f}")

    res = pd.DataFrame(gs.cv_results_)
    res.drop(columns=[c for c in res.columns if c.startswith("param_")]).assign(
        params=res["params"].astype(str)
    ).sort_values("rank_test_f1_macro").to_csv(C.CV_DIR / f"cv_results_{name}.csv", index=False)

    i = gs.best_index_
    row = {"model": name, "best_params": json.dumps(gs.best_params_)}
    for m in SCORING:
        row[f"cv_{m}_mean"] = res.loc[i, f"mean_test_{m}"]
        row[f"cv_{m}_std"] = res.loc[i, f"std_test_{m}"]
    row["mean_fit_time_s"] = res.loc[i, "mean_fit_time"]
    row["search_time_min"] = t_search / 60
    row["n_fits"] = n_combos * C.CV_FOLDS

    w = grid_edge_warnings(name, gs.best_params_, spec["grid"])
    for msg in w:
        print("GRID-EDGE WARNING:", msg)
    all_warnings += w

    # ---- final pipeline (what gets shipped) ----
    final = final_estimator(name, gs.best_estimator_)

    # per-class CV F1 of the shipped configuration, same folds
    t0 = time.time()
    y_cv = cross_val_predict(final, X, y, cv=CV, n_jobs=1)
    print(f"{name}: cross_val_predict done in {time.time() - t0:.0f}s")
    rep = classification_report(y, y_cv, labels=C.CLASS_ORDER, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(C.CV_DIR / f"cv_per_class_report_{name}.csv")
    pd.DataFrame(confusion_matrix(y, y_cv, labels=C.CLASS_ORDER), index=C.CLASS_ORDER,
                 columns=C.CLASS_ORDER).to_csv(C.CV_DIR / f"cv_confusion_{name}.csv")
    row["cvpred_f1_macro_final_model"] = f1_score(y, y_cv, average="macro")
    per_class_rows.append({"model": name, **{c: rep[c]["f1-score"] for c in C.CLASS_ORDER}})
    print(f"{name}: per-class CV F1 " + ", ".join(f"{c}={rep[c]['f1-score']:.3f}" for c in C.CLASS_ORDER))

    # refit on full training set
    t0 = time.time(); final.fit(X, y); row["final_refit_s"] = time.time() - t0

    # consistency check: predict == argmax(predict_proba)  (on training data)
    proba = final.predict_proba(X)
    pred = final.predict(X)
    argmax_lab = final.classes_[proba.argmax(axis=1)]
    n_mismatch = int((pred != argmax_lab).sum())
    row["predict_vs_argmax_mismatches"] = n_mismatch
    print(f"{name}: predict == argmax(predict_proba): {n_mismatch} mismatches / {len(pred)}")
    assert list(final.classes_) == C.CLASS_ORDER, final.classes_

    path = C.MODELS_DIR / f"{name}_pipeline.joblib"
    joblib.dump(final, path, compress=3)
    row["model_file_mb"] = path.stat().st_size / 1e6
    print(f"{name}: saved {path.name} ({row['model_file_mb']:.1f} MB)")

    used = list(final[:-1].get_feature_names_out())
    (C.MODELS_DIR / f"{name}_used_genes.json").write_text(json.dumps(used, indent=0))
    row["n_used_genes"] = len(used)
    summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(C.CV_DIR / "cv_summary.csv", index=False)  # incremental

# ------------------------------------------------------------------ shared metadata
(C.MODELS_DIR / "feature_genes.json").write_text(json.dumps(feats))
(C.MODELS_DIR / "class_labels.json").write_text(json.dumps(C.CLASS_ORDER))
pd.DataFrame(per_class_rows).set_index("model").to_csv(C.CV_DIR / "cv_per_class_f1.csv")
(C.CV_DIR / "grid_edge_warnings.txt").write_text("\n".join(all_warnings) or "none\n")
meta = {
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "sklearn": sklearn.__version__, "n_train": int(len(X)), "n_input_genes": len(feats),
    "cv": f"StratifiedKFold({C.CV_FOLDS}, shuffle=True, random_state={C.RANDOM_STATE})",
    "class_order": C.CLASS_ORDER,
}
(C.MODELS_DIR / "training_metadata.json").write_text(json.dumps(meta, indent=2))

shutil.rmtree(PIPELINE_CACHE, ignore_errors=True)
print(f"\nPipeline cache cleared ({PIPELINE_CACHE})")
print(f"PHASE 3 DONE in {(time.time() - T_START) / 60:.1f} min")
print("\nCV SUMMARY\n" + pd.DataFrame(summary_rows).to_string(index=False))
print("\nPER-CLASS CV F1\n" + pd.DataFrame(per_class_rows).to_string(index=False))
print("\nGRID-EDGE WARNINGS\n" + ("\n".join(all_warnings) or "none"))
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
log_fh.close()
