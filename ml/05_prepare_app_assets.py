"""Phase 6 build step: precompute small assets for the Streamlit app and verify the
"only used genes" input. The app itself never loads the dataset.

Creates:
  models/train_gene_stats.csv.gz        training-set median / p01 / p99 / min / max per gene
  models/lr_shap_background_mean.npy    mean of the LR SHAP background (same 300 train samples as Phase 5)
  app/sample_input/demo_samples.csv     10 test samples (2 per subtype), labels removed
  results/ml/app_checks/                demo labels + verification results

Models are only loaded, never changed. Run from repo root:  python ml/05_prepare_app_assets.py
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap

from ml import config as C
from ml.data import load_split
from ml.inference import (MODEL_NAMES, build_model_input, load_used_genes, lr_contributions,
                          parse_expression_table, predict)

CHECK_DIR = C.ML_RESULTS / "app_checks"
CHECK_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR = C.APP_DIR / "sample_input"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

data, feats = load_split(which=("train", "test"))
X_tr, y_tr = data["train"]
X_te, y_te = data["test"]

# ---- 1. training-set gene statistics (for median fill + input sanity checks)
q = X_tr.quantile([0.01, 0.5, 0.99]).T
stats = pd.DataFrame({"median": q[0.5], "p01": q[0.01], "p99": q[0.99],
                      "min": X_tr.min(), "max": X_tr.max()}).astype(np.float32)
stats.index.name = "gene"
stats.to_csv(C.MODELS_DIR / "train_gene_stats.csv.gz", float_format="%.5f")
print(f"gene stats: {stats.shape}, value range {stats['min'].min():.2f}..{stats['max'].max():.2f}")

# ---- 2. LR SHAP background mean (identical draw to ml/04_explain.py)
lr = joblib.load(C.MODELS_DIR / "logreg_pipeline.joblib")
rng = np.random.default_rng(C.RANDOM_STATE)
Z_tr = lr[:-1].transform(X_tr).astype(np.float64)
bg_idx = rng.choice(len(Z_tr), size=300, replace=False)
# NOTE: shap's default Independent masker subsamples the background to 100 rows, so the
# effective baseline is the mean of those 100 rows. Use shap's own mean so the app matches
# the Phase 5 SHAP values exactly.
expl = shap.LinearExplainer(lr.named_steps["model"], Z_tr[bg_idx])
bg_mean = np.asarray(expl.mean, dtype=np.float64)
print(f"shap background rows used: {expl.masker.data.shape[0]} of {len(bg_idx)}")
np.save(C.MODELS_DIR / "lr_shap_background_mean.npy", bg_mean)
# verify the closed-form contributions equal shap.LinearExplainer
Zt = lr[:-1].transform(X_te.iloc[:50]).astype(np.float64)
ref = np.asarray(expl.shap_values(Zt))
phi, base, _ = lr_contributions(lr, X_te.iloc[:50], bg_mean)
shap_err = float(np.abs(phi - ref).max())
base_err = float(np.abs(base - np.asarray(expl.expected_value)).max())
print(f"LR contributions vs shap.LinearExplainer: max |diff| {shap_err:.2e}, base diff {base_err:.2e}")
assert shap_err < 1e-8 and base_err < 1e-8

# ---- 3. demo samples: 2 random test samples per subtype, labels removed
demo_ids = (y_te.groupby(y_te, group_keys=False)
            .apply(lambda s: s.sample(2, random_state=C.RANDOM_STATE)).index.tolist())
demo = X_te.loc[demo_ids]
demo.index.name = "sample_id"
demo.to_csv(DEMO_DIR / "demo_samples.csv", float_format="%.6g")
y_te.loc[demo_ids].rename("true_subtype").to_csv(CHECK_DIR / "demo_labels.csv")
print(f"demo: {len(demo)} samples -> {DEMO_DIR / 'demo_samples.csv'} "
      f"({(DEMO_DIR / 'demo_samples.csv').stat().st_size / 1e6:.1f} MB)")

# ---- 4. verification: full 30,865-gene input vs "only used genes" input (as the app does it)
gene_set = set(feats)
raw = pd.read_csv(DEMO_DIR / "demo_samples.csv")
parsed = parse_expression_table(raw, gene_set)
assert parsed.data is not None and parsed.data.shape == (10, len(feats))
full_df = parsed.data[feats].astype(np.float32)
labels = y_te.loc[demo_ids]

# transposed file must parse to the same matrix
buf = io.StringIO(); parsed.data.T.to_csv(buf); buf.seek(0)
parsed_t = parse_expression_table(pd.read_csv(buf), gene_set)
transposed_ok = parsed_t.data is not None and np.array_equal(
    parsed_t.data[feats].to_numpy(np.float32), full_df.to_numpy())

rows = []
for m in MODEL_NAMES:
    pipe = joblib.load(C.MODELS_DIR / f"{m}_pipeline.joblib")
    used = load_used_genes(m)
    pred_full, p_full, _ = predict(pipe, full_df)
    mi = build_model_input(parsed.data[used], used, feats, stats["median"])  # only used genes uploaded
    assert mi.X is not None
    pred_red, p_red, _ = predict(pipe, mi.X)
    # the features reaching the final estimator must be bitwise identical
    feats_identical = bool(np.array_equal(pipe[:-1].transform(full_df), pipe[:-1].transform(mi.X)))
    # run-to-run noise of the model itself on the SAME input (RF predicts with 2 threads)
    _, p_full2, _ = predict(pipe, full_df)
    self_noise = float(np.abs(p_full - p_full2).max())
    # refusal check: drop one used gene
    refused = build_model_input(parsed.data[used[1:]], used, feats, stats["median"]).X is None
    rows.append({
        "model": m, "n_used_genes": len(used),
        "model_features_bitwise_identical": feats_identical,
        "predictions_identical": bool((pred_full == pred_red).all()),
        "probabilities_bitwise_identical": bool(np.array_equal(p_full, p_red)),
        "max_abs_prob_diff": float(np.abs(p_full - p_red).max()),
        "same_input_rerun_prob_diff": self_noise,
        "refuses_when_used_gene_missing": refused,
        "transposed_input_parsed_identically": transposed_ok,
        "demo_correct": int((pred_full == labels.to_numpy()).sum()), "demo_n": len(labels),
    })
    pd.DataFrame({"sample_id": demo_ids, "true": labels.values, "pred_full": pred_full, "pred_used_only": pred_red,
                  **{f"p_full_{c}": p_full[:, k] for k, c in enumerate(C.CLASS_ORDER)}}) \
        .to_csv(CHECK_DIR / f"demo_predictions_{m}.csv", index=False)
chk = pd.DataFrame(rows)
chk.to_csv(CHECK_DIR / "full_vs_used_genes_check.csv", index=False)
pd.set_option("display.width", 250)
print("\n" + chk.to_string(index=False))
assert chk["model_features_bitwise_identical"].all() and chk["predictions_identical"].all()
assert (chk["max_abs_prob_diff"] < 1e-12).all()  # bitwise for LR/SVC; RF only float-summation noise
assert chk["refuses_when_used_gene_missing"].all() and transposed_ok
print("\nAll app-input checks passed.")
