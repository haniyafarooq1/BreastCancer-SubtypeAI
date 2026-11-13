"""Inference helpers shared by the Streamlit app and ml/05_prepare_app_assets.py.

"Only used genes" input: a model needs only the genes that survive its feature
selection (500-1000). All other input columns are filled with the training-set median.
They are dropped inside the pipeline, so predictions are identical to a full
30,865-gene input (verified in ml/05_prepare_app_assets.py).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_NAMES = {"logreg": "Logistic Regression (primary)", "svc": "SVC (RBF, calibrated)",
               "rf": "Random Forest"}
MAX_SAMPLES = 500
ID_ALIASES = ("sample_id", "sampleid", "sample", "id", "")


# ------------------------------------------------------------------ assets
def load_feature_genes():
    return json.loads((MODELS_DIR / "feature_genes.json").read_text())


def load_class_labels():
    return json.loads((MODELS_DIR / "class_labels.json").read_text())


def load_used_genes(model):
    return json.loads((MODELS_DIR / f"{model}_used_genes.json").read_text())


def load_gene_stats():
    """Training-set per-gene median and 1st/99th percentiles (all 30,865 genes)."""
    return pd.read_csv(MODELS_DIR / "train_gene_stats.csv.gz", index_col="gene")


def sklearn_version_check():
    """(ok, installed, trained_with). Pickled sklearn models are only guaranteed to load and
    behave identically with the scikit-learn version they were trained with."""
    import sklearn
    meta_path = MODELS_DIR / "training_metadata.json"
    trained = json.loads(meta_path.read_text()).get("sklearn") if meta_path.exists() else None
    return (trained is None or sklearn.__version__ == trained), sklearn.__version__, trained


def load_pipeline(model):
    import warnings
    import joblib
    ok, inst, trained = sklearn_version_check()
    if not ok:
        warnings.warn(f"scikit-learn {inst} is installed but the models were trained with {trained}. "
                      f"Install the pinned version: pip install -r requirements.txt", stacklevel=2)
    return joblib.load(MODELS_DIR / f"{model}_pipeline.joblib")


# ------------------------------------------------------------------ parsing
@dataclass
class ParsedInput:
    data: pd.DataFrame | None                      # samples x genes (numeric), index = sample id
    messages: list = field(default_factory=list)   # (level, text) with level in info/warning/error


def _gene_hits(values, gene_set):
    return sum(1 for v in values if str(v).strip() in gene_set)


def parse_expression_table(raw: pd.DataFrame, gene_set: set) -> ParsedInput:
    """Accept samples-as-rows (optional sample_id column) or genes-as-rows (auto-transposed)."""
    msgs = []
    if raw.empty:
        return ParsedInput(None, [("error", "The file is empty.")])
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    first = raw.columns[0]
    col_hits = _gene_hits(raw.columns, gene_set)
    row_hits = _gene_hits(raw[first].astype(str), gene_set)

    if row_hits > col_hits:  # genes as rows -> transpose
        df = raw.set_index(first)
        df.index = df.index.astype(str).str.strip()
        df = df.T
        msgs.append(("info", f"Detected genes as rows ({row_hits} known genes in the first column); "
                             f"the table was transposed to samples as rows."))
    else:
        id_col = next((c for c in raw.columns if c.lower().replace(" ", "") in ID_ALIASES
                       or c.lower().startswith("unnamed")), None)
        if id_col is None and not pd.api.types.is_numeric_dtype(raw[first]) and first not in gene_set:
            id_col = first
        if id_col is not None:
            df = raw.set_index(id_col)
        else:
            df = raw
            df.index = [f"sample_{i + 1}" for i in range(len(df))]
    df.index = df.index.astype(str)
    df.index.name = "sample_id"
    df.columns = [str(c).strip() for c in df.columns]

    if df.columns.duplicated().any():
        dups = sorted(set(df.columns[df.columns.duplicated()]))
        msgs.append(("warning", f"Duplicate gene columns found; keeping the first occurrence: "
                                f"{', '.join(dups[:10])}{' ...' if len(dups) > 10 else ''}"))
        df = df.loc[:, ~df.columns.duplicated()]
    if len(df) > MAX_SAMPLES:
        msgs.append(("warning", f"{len(df)} samples uploaded; only the first {MAX_SAMPLES} are used."))
        df = df.iloc[:MAX_SAMPLES]
    if len(gene_set & set(df.columns)) == 0:
        msgs.append(("error", "No known gene names were found in the columns or the first column. "
                              "Use HGNC gene symbols as in the template (e.g. ESR1, ERBB2)."))
        return ParsedInput(None, msgs)
    return ParsedInput(df, msgs)


# ------------------------------------------------------------------ model input
@dataclass
class ModelInput:
    X: pd.DataFrame | None
    missing_used: list
    extra_cols: list
    messages: list


def build_model_input(df: pd.DataFrame, used_genes, feature_genes, medians: pd.Series) -> ModelInput:
    """Full 30,865-column matrix: used genes from the upload, everything else = training median.
    Refuses (X=None) if any used gene is missing or non-numeric."""
    msgs = []
    used_set = set(used_genes)
    missing = [g for g in used_genes if g not in df.columns]
    extra = [c for c in df.columns if c not in used_set]
    if missing:
        msgs.append(("error", f"{len(missing)} of the {len(used_genes)} genes required by this model "
                              f"are missing. Prediction refused."))
        return ModelInput(None, missing, extra, msgs)

    vals = df[used_genes].apply(pd.to_numeric, errors="coerce")
    bad = vals.columns[vals.isna().any()].tolist()
    if bad:
        msgs.append(("error", f"{len(bad)} required genes contain empty or non-numeric values "
                              f"(e.g. {', '.join(bad[:10])}). Prediction refused."))
        return ModelInput(None, bad, extra, msgs)
    if extra:
        msgs.append(("info", f"{len(extra)} extra columns are not used by this model and were ignored "
                             f"(they are replaced by training-set medians, which do not affect the "
                             f"prediction)."))

    base = medians.reindex(feature_genes).to_numpy(np.float32)
    X = pd.DataFrame(np.tile(base, (len(df), 1)), index=df.index, columns=feature_genes)
    X[used_genes] = vals.to_numpy(np.float32)
    return ModelInput(X, [], extra, msgs)


def sanity_check(values: pd.DataFrame, stats: pd.DataFrame):
    """Warn if values do not look like log2(FPKM + 0.1) on the training scale."""
    warns = []
    v = values.to_numpy(np.float64)
    st = stats.loc[values.columns]
    # log2(FPKM + 0.1) data has many negative values (floor -3.32); counts / unlogged data have ~none
    if float((v < -0.5).mean()) < 0.001 and np.nanmax(v) > 50:
        warns.append("Values are (almost) all non-negative and some exceed 50: this looks like raw counts or "
                     "unlogged FPKM/TPM. The models expect log2(FPKM + 0.1) values (training range "
                     "roughly -3.3 to 15).")
    if np.nanmax(v) > 25:
        warns.append(f"Maximum value {np.nanmax(v):.1f} is far above the training range "
                     f"(max ≈ {stats['max'].max():.1f}); check the data scale.")
    lo = st["p01"].to_numpy() - 2.0
    hi = st["p99"].to_numpy() + 2.0
    frac_out = float(((v < lo) | (v > hi)).mean())
    if frac_out > 0.10:
        warns.append(f"{frac_out:.0%} of the values lie more than 2 log2 units outside the training "
                     f"1st–99th percentile range of their gene. The data may be on a different scale "
                     f"or from a different platform/normalisation; predictions may be unreliable.")
    shift = float(np.nanmedian(v - st["median"].to_numpy()))
    if abs(shift) > 2:
        warns.append(f"Values are shifted by a median of {shift:+.1f} log2 units relative to the training "
                     f"medians of the same genes; check normalisation.")
    return warns, frac_out


def predict(pipe, X):
    proba = pipe.predict_proba(X)
    classes = np.asarray(pipe.classes_)
    return classes[proba.argmax(axis=1)], proba, classes


def lr_contributions(pipe, X, background_mean):
    """Exact SHAP values of the LR model (LinearExplainer, interventional):
    phi[i, j, c] = coef[c, j] * (z[i, j] - mean_bg[j]), on the log-odds scale."""
    Z = pipe[:-1].transform(X).astype(np.float64)
    coef = pipe.named_steps["model"].coef_  # (classes, genes)
    phi = (Z - background_mean)[:, :, None] * coef.T[None, :, :]
    genes = np.asarray(pipe[:-1].get_feature_names_out())
    base = pipe.named_steps["model"].intercept_ + coef @ background_mean
    return phi, base, genes
