"""Phase 4: ONE-TIME evaluation of the three saved pipelines on the held-out test set.

No fitting happens here: pipelines are only loaded from models/. The primary model is
chosen by the pre-registered rule in results/ml/model_selection_rule.md (CV-based),
NOT by the test results below.

A guard refuses to run again once test results exist (use --force to override).
Run from repo root:  python ml/03_evaluate.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_recall_fscore_support)

from ml import config as C
from ml.data import load_split

MODELS = {"logreg": "Logistic Regression", "svc": "SVC (RBF, calibrated)", "rf": "Random Forest"}
PRIMARY = "logreg"  # fixed by results/ml/model_selection_rule.md before this script was run
C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
GUARD = C.EVAL_DIR / "test_metrics.csv"
if GUARD.exists() and "--force" not in sys.argv:
    sys.exit(f"Test set already evaluated ({GUARD}). Refusing to re-run; pass --force to override.")

# Sequential one-hue blue ramp (light -> dark)
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fbff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED = "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 11})

data, feats = load_split(which=("train", "test"))
X_tr, y_tr = data["train"]
X_te, y_te = data["test"]
print(f"X_test {X_te.shape} | test classes: {y_te.value_counts().reindex(C.CLASS_ORDER).to_dict()}")
cv = pd.read_csv(C.CV_DIR / "cv_summary.csv").set_index("model")


def plot_confusion(cm, name, title):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, mat, fmt, sub in ((axes[0], cm, "d", "Counts"),
                              (axes[1], cm_norm, ".2f", "Normalized by true class (recall)")):
        im = ax.imshow(mat, cmap=BLUES, vmin=0, vmax=mat.max() if fmt == "d" else 1)
        ax.set_xticks(range(5), C.CLASS_ORDER); ax.set_yticks(range(5), C.CLASS_ORDER)
        ax.set_xlabel("Predicted subtype"); ax.set_ylabel("True subtype"); ax.set_title(sub, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        thresh = (mat.max() if fmt == "d" else 1) * 0.55
        for i in range(5):
            for j in range(5):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > thresh else INK)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.outline.set_visible(False)
    fig.suptitle(f"{title}: test-set confusion matrix (n={cm.sum()})", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.savefig(C.EVAL_DIR / f"confusion_matrix_{name}.png", dpi=200)
    plt.close(fig)


rows, per_class_f1, confusions = [], {}, {}
for name, title in MODELS.items():
    pipe = joblib.load(C.MODELS_DIR / f"{name}_pipeline.joblib")
    assert list(pipe.classes_) == C.CLASS_ORDER
    proba = pipe.predict_proba(X_te)
    pred = pipe.predict(X_te)
    assert (pred == pipe.classes_[proba.argmax(1)]).all()

    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    train_f1 = f1_score(y_tr, pipe.predict(X_tr), average="macro")  # resubstitution (fit) score
    rows.append({
        "model": name, "model_name": title, "primary_by_cv_rule": name == PRIMARY,
        "test_accuracy": accuracy_score(y_te, pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "test_precision_macro": pr_m, "test_recall_macro": rc_m, "test_f1_macro": f1_m,
        "test_precision_weighted": pr_w, "test_recall_weighted": rc_w, "test_f1_weighted": f1_w,
        "test_log_loss": log_loss(y_te, proba, labels=C.CLASS_ORDER),
        "cv_f1_macro_mean": cv.loc[name, "cv_f1_macro_mean"], "cv_f1_macro_std": cv.loc[name, "cv_f1_macro_std"],
        "test_minus_cv_f1_macro": f1_m - cv.loc[name, "cv_f1_macro_mean"],
        "train_fit_f1_macro": train_f1,
        "best_params": cv.loc[name, "best_params"],
    })
    rep = classification_report(y_te, pred, labels=C.CLASS_ORDER, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(C.EVAL_DIR / f"classification_report_{name}.csv")
    per_class_f1[name] = {c: rep[c]["f1-score"] for c in C.CLASS_ORDER}
    print(f"\n--- {title} ---\n" + classification_report(y_te, pred, labels=C.CLASS_ORDER, digits=3))

    cm = confusion_matrix(y_te, pred, labels=C.CLASS_ORDER)
    confusions[name] = cm
    pd.DataFrame(cm, index=C.CLASS_ORDER, columns=C.CLASS_ORDER).to_csv(C.EVAL_DIR / f"confusion_counts_{name}.csv")
    pd.DataFrame(cm / cm.sum(1, keepdims=True), index=C.CLASS_ORDER, columns=C.CLASS_ORDER).round(4) \
        .to_csv(C.EVAL_DIR / f"confusion_normalized_{name}.csv")
    plot_confusion(cm, name, title)

    out = pd.DataFrame(proba, index=X_te.index, columns=[f"prob_{c}" for c in C.CLASS_ORDER])
    out.insert(0, "predicted", pred); out.insert(0, "true", y_te.values)
    out.to_csv(C.EVAL_DIR / f"test_predictions_{name}.csv")

comp = pd.DataFrame(rows)
comp.to_csv(GUARD, index=False)
comp.to_csv(C.EVAL_DIR / "model_comparison.csv", index=False)
pcf = pd.DataFrame(per_class_f1).T[C.CLASS_ORDER]
pcf.to_csv(C.EVAL_DIR / "test_per_class_f1.csv")
cvpc = pd.read_csv(C.CV_DIR / "cv_per_class_f1.csv").set_index("model")[C.CLASS_ORDER]

# Main confusion pairs (true -> predicted)
PAIRS = [("Normal", "LumA"), ("LumA", "Normal"), ("LumA", "LumB"), ("LumB", "LumA"), ("Her2", "LumB"),
         ("LumB", "Her2")]
idx = {c: i for i, c in enumerate(C.CLASS_ORDER)}
conf_rows = []
for a, b in PAIRS:
    r = {"true -> predicted": f"{a} -> {b}"}
    for name in MODELS:
        cm = confusions[name]
        n, tot = cm[idx[a], idx[b]], cm[idx[a]].sum()
        r[name] = f"{n}/{tot} ({n / tot:.0%})"
    conf_rows.append(r)
conf_df = pd.DataFrame(conf_rows)
conf_df.to_csv(C.EVAL_DIR / "main_confusions.csv", index=False)

(C.MODELS_DIR / "best_model.json").write_text(json.dumps({
    "primary_model": PRIMARY, "file": f"{PRIMARY}_pipeline.joblib",
    "selected_by": "results/ml/model_selection_rule.md (CV macro F1; LR tie-break within 1 SD)",
    "class_order": C.CLASS_ORDER,
}, indent=2))


def md_table(df, floatfmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(floatfmt.format)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


show = comp[["model_name", "test_accuracy", "test_balanced_accuracy", "test_precision_macro",
             "test_recall_macro", "test_f1_macro", "test_f1_weighted", "test_log_loss"]]
over = comp[["model_name", "train_fit_f1_macro", "cv_f1_macro_mean", "cv_f1_macro_std", "test_f1_macro",
             "test_minus_cv_f1_macro"]]
pc = pd.concat({"CV": cvpc, "Test": pcf}, axis=1)
pc.columns = [f"{s} {c}" for s, c in pc.columns]
md = [
    "# Phase 4: test-set evaluation", "",
    f"Generated {datetime.now():%Y-%m-%d %H:%M}. Test set: {len(y_te)} samples, evaluated once. "
    f"Primary model (chosen by CV rule before testing): **{MODELS[PRIMARY]}**.", "",
    "## Test metrics", md_table(show), "",
    "## Overfitting check (macro F1)", md_table(over), "",
    "## Per-class F1: CV vs test", md_table(pc.reset_index().rename(columns={"index": "model"})), "",
    "## Main confusions on the test set (true -> predicted, share of the true class)", md_table(conf_df), "",
]
(C.EVAL_DIR / "evaluation_summary.md").write_text("\n".join(md), encoding="utf-8")

pd.set_option("display.width", 250)
print("\n" + show.round(4).to_string(index=False))
print("\n" + over.round(4).to_string(index=False))
print("\n" + pc.round(3).to_string())
print("\n" + conf_df.to_string(index=False))
