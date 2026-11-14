"""Shared training helpers (same logic as ml/02_train_cv.py), used by ml/02b_grid_extension.py."""
import json
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict

from ml import config as C

SCORING = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}


class TimestampTee:
    """Duplicate a stream to a log file, prefixing each line with a timestamp."""
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


def start_log(path):
    fh = open(path, "a", encoding="utf-8")
    sys.stdout = TimestampTee(sys.__stdout__, fh)
    sys.stderr = TimestampTee(sys.__stderr__, fh)
    return fh


def stop_log(fh):
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    fh.close()


def grid_edge_warnings(name, best, grid):
    """Flag C / k whose best value is at the edge of the sub-grid containing the best params."""
    warns = []
    sub = next(g for g in grid if all(best.get(p) in v for p, v in g.items()))
    for p in ("model__C", "select__k"):
        if p not in sub or len(sub[p]) < 2:
            continue
        vals, b = sorted(sub[p]), best[p]
        if b in (vals[0], vals[-1]):
            side = "lower" if b == vals[0] else "upper"
            warns.append(f"{name}: best {p}={b} is at the {side} edge of {vals} (not extended further)")
    return warns


def final_estimator(name, best_pipe):
    pipe = clone(best_pipe)
    pipe.set_params(memory=None)
    if name == "svc":
        pipe.set_params(model=CalibratedClassifierCV(pipe.named_steps["model"], cv=5, method="sigmoid"))
    return pipe


def finalize_model(name, best_pipe, X, y, cv):
    """Per-class CV F1 via cross_val_predict, refit on full train, consistency check, save.
    Returns (summary-fields dict, per-class F1 dict)."""
    final = final_estimator(name, best_pipe)
    t0 = time.time()
    y_cv = cross_val_predict(final, X, y, cv=cv, n_jobs=1)
    print(f"{name}: cross_val_predict done in {time.time() - t0:.0f}s")
    rep = classification_report(y, y_cv, labels=C.CLASS_ORDER, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(C.CV_DIR / f"cv_per_class_report_{name}.csv")
    pd.DataFrame(confusion_matrix(y, y_cv, labels=C.CLASS_ORDER), index=C.CLASS_ORDER,
                 columns=C.CLASS_ORDER).to_csv(C.CV_DIR / f"cv_confusion_{name}.csv")
    row = {"cvpred_f1_macro_final_model": f1_score(y, y_cv, average="macro")}
    per_class = {c: rep[c]["f1-score"] for c in C.CLASS_ORDER}
    print(f"{name}: per-class CV F1 " + ", ".join(f"{c}={v:.3f}" for c, v in per_class.items()))

    t0 = time.time(); final.fit(X, y); row["final_refit_s"] = time.time() - t0
    proba, pred = final.predict_proba(X), final.predict(X)
    row["predict_vs_argmax_mismatches"] = int((pred != final.classes_[proba.argmax(axis=1)]).sum())
    print(f"{name}: predict == argmax(predict_proba): {row['predict_vs_argmax_mismatches']} mismatches")
    assert list(final.classes_) == C.CLASS_ORDER

    path = C.MODELS_DIR / f"{name}_pipeline.joblib"
    joblib.dump(final, path, compress=3)
    row["model_file_mb"] = path.stat().st_size / 1e6
    used = list(final[:-1].get_feature_names_out())
    (C.MODELS_DIR / f"{name}_used_genes.json").write_text(json.dumps(used, indent=0))
    row["n_used_genes"] = len(used)
    print(f"{name}: saved {path.name} ({row['model_file_mb']:.1f} MB, {len(used)} genes)")
    return row, per_class
