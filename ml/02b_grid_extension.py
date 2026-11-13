"""Phase 3b: grid extensions requested after the Phase 3 grid-edge warnings.
  SVC: add C=100 (rbf) and k=250;  RF: add k=250.
Only the NEW combinations are evaluated (same folds/scoring), merged with the Phase 3
cv_results, and re-ranked. Pipelines are re-saved only if the best params change.
Training set only. Run from repo root:  python ml/02b_grid_extension.py
"""
import ast
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from ml import config as C
from ml.data import load_split
from ml.pipeline import PIPELINE_CACHE, build_pipeline
from ml.training import SCORING, finalize_model, grid_edge_warnings, start_log, stop_log

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("default", ConvergenceWarning)
log_fh = start_log(C.CV_DIR / "train_log.txt")
T0 = time.time()
print(f"\n{'#' * 70}\nPHASE 3b GRID EXTENSION RUN\n{'#' * 70}")

data, _ = load_split(which=("train",))
X, y = data["train"]
print(f"X_train {X.shape}; test set NOT loaded")
CV = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
shutil.rmtree(PIPELINE_CACHE, ignore_errors=True)

SPECS = {
    "svc": dict(
        estimator=SVC(class_weight="balanced", random_state=C.RANDOM_STATE),
        new_grid=[  # only combinations not run in Phase 3
            {"select__k": [250], "model__kernel": ["linear"], "model__C": [0.001, 0.01, 0.1]},
            {"select__k": [250], "model__kernel": ["rbf"], "model__C": [1, 10], "model__gamma": ["scale"]},
            {"select__k": [250, 500, 1000, 2000], "model__kernel": ["rbf"], "model__C": [100],
             "model__gamma": ["scale"]},
        ],
        full_grid=[  # Phase 3 + extension, used for the edge check
            {"select__k": [250, 500, 1000, 2000], "model__kernel": ["linear"], "model__C": [0.001, 0.01, 0.1]},
            {"select__k": [250, 500, 1000, 2000], "model__kernel": ["rbf"], "model__C": [1, 10, 100],
             "model__gamma": ["scale"]},
        ],
    ),
    "rf": dict(
        estimator=RandomForestClassifier(n_estimators=300, max_features="sqrt", max_depth=None,
                                         class_weight="balanced", n_jobs=2, random_state=C.RANDOM_STATE),
        new_grid=[{"select__k": [250], "model__min_samples_leaf": [1, 3]}],
        full_grid=[{"select__k": [250, 500, 1000, 2000], "model__min_samples_leaf": [1, 3]}],
    ),
}

summary = pd.read_csv(C.CV_DIR / "cv_summary.csv").set_index("model")
per_class = pd.read_csv(C.CV_DIR / "cv_per_class_f1.csv").set_index("model")
warnings_all = {"logreg": []}  # logreg had no edge warnings in Phase 3 and is not re-run

for name, spec in SPECS.items():
    print(f"\n{'=' * 70}\nEXTENSION: {name}\n{'=' * 70}")
    old = pd.read_csv(C.CV_DIR / f"cv_results_{name}.csv")
    old_best = ast.literal_eval(old.sort_values("rank_test_f1_macro").iloc[0]["params"])

    gs = GridSearchCV(build_pipeline(spec["estimator"]), spec["new_grid"], scoring=SCORING,
                      refit=False, cv=CV, n_jobs=1, pre_dispatch="n_jobs", verbose=3,
                      error_score="raise")
    t0 = time.time(); gs.fit(X, y); t_ext = time.time() - t0
    new = pd.DataFrame(gs.cv_results_)
    new = new.drop(columns=[c for c in new.columns if c.startswith("param_")])
    new["params"] = new["params"].astype(str)
    print(f"{name}: {len(new)} new combos ({len(new) * C.CV_FOLDS} fits) in {t_ext / 60:.1f} min")

    # merge + re-rank on all metrics
    allres = pd.concat([old, new], ignore_index=True)
    for m in SCORING:
        allres[f"rank_test_{m}"] = allres[f"mean_test_{m}"].rank(ascending=False, method="min").astype(int)
    allres = allres.sort_values("rank_test_f1_macro").reset_index(drop=True)
    allres.to_csv(C.CV_DIR / f"cv_results_{name}.csv", index=False)
    best_row = allres.iloc[0]
    best = ast.literal_eval(best_row["params"])
    print(f"{name}: previous best {old_best} | new best {best} "
          f"(macro F1 {best_row['mean_test_f1_macro']:.4f} ± {best_row['std_test_f1_macro']:.4f})")
    print("top 5 combos:\n" + allres[["params", "mean_test_f1_macro", "std_test_f1_macro"]]
          .head(5).to_string(index=False))

    w = grid_edge_warnings(name, best, spec["full_grid"])
    for msg in w:
        print("GRID-EDGE WARNING:", msg)
    warnings_all[name] = w

    s = summary.loc[name].to_dict()
    s.update({
        "best_params": json.dumps(best),
        **{f"cv_{m}_mean": best_row[f"mean_test_{m}"] for m in SCORING},
        **{f"cv_{m}_std": best_row[f"std_test_{m}"] for m in SCORING},
        "mean_fit_time_s": best_row["mean_fit_time"],
        "search_time_min": s["search_time_min"] + t_ext / 60,
        "n_fits": int(s["n_fits"]) + len(new) * C.CV_FOLDS,
    })
    if best != old_best:
        print(f"{name}: BEST PARAMS CHANGED -> refitting and re-saving the final pipeline")
        pipe = build_pipeline(spec["estimator"], memory=None).set_params(**best)
        row, pc = finalize_model(name, pipe, X, y, CV)
        s.update(row)
        per_class.loc[name] = pd.Series(pc)
    else:
        print(f"{name}: best params unchanged -> saved pipeline kept as is")
    summary.loc[name] = pd.Series(s)

summary.reset_index().to_csv(C.CV_DIR / "cv_summary.csv", index=False)
per_class.to_csv(C.CV_DIR / "cv_per_class_f1.csv")
lines = [m for v in warnings_all.values() for m in v]
(C.CV_DIR / "grid_edge_warnings.txt").write_text(
    "After Phase 3b grid extension (final; no further extension):\n" + ("\n".join(lines) or "none") + "\n")

shutil.rmtree(PIPELINE_CACHE, ignore_errors=True)
print(f"\nPipeline cache cleared. PHASE 3b DONE in {(time.time() - T0) / 60:.1f} min")
pd.set_option("display.width", 250)
print("\nCV SUMMARY\n" + summary[["best_params", "cv_f1_macro_mean", "cv_f1_macro_std",
                                   "cv_balanced_accuracy_mean", "cv_balanced_accuracy_std"]].to_string())
print("\nPER-CLASS CV F1\n" + per_class.round(3).to_string())
print("\nGRID-EDGE WARNINGS\n" + ("\n".join(lines) or "none"))
stop_log(log_fh)
