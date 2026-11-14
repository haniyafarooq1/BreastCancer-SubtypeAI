"""Shared helpers for all app pages: paths, disclaimer, cached loaders, safe image display.

The app only reads saved models, JSON files and pre-computed results; it never loads
the expression dataset.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # needed to unpickle the pipelines (ml.transformers)
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from ml import inference as inf

RESULTS = ROOT / "results"
ML = RESULTS / "ml"
EVAL = ML / "evaluation"
CV = ML / "cv"
EXPL = ML / "explainability"
DEMO_FILE = ROOT / "app" / "sample_input" / "demo_samples.csv"

DISCLAIMER = "Research/educational tool. Not for clinical diagnosis."
CLASS_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]
SUBTYPE_COLORS = dict(zip(CLASS_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
TOWARDS, AWAY = "#e34948", "#2a78d6"
MODEL_NAMES = inf.MODEL_NAMES


def setup_page(title, icon="🧬"):
    st.set_page_config(page_title=f"{title} · BreastCancer-SubtypeAI", page_icon=icon, layout="wide")
    st.sidebar.warning(f"⚠️ {DISCLAIMER}")
    ok, inst, trained = inf.sklearn_version_check()
    if not ok:
        st.sidebar.error(f"**Version mismatch:** scikit-learn {inst} is installed, but the models were "
                         f"trained with {trained}. Predictions may fail or differ. Run "
                         f"`pip install -r requirements.txt` in your virtual environment.")
    st.sidebar.caption("BreastCancer-SubtypeAI · GSE96058 (SCAN-B) · PAM50 subtypes")


def footer():
    st.divider()
    st.caption(f"⚠️ {DISCLAIMER} Models were trained on a single cohort (SCAN-B) and have not been "
               "externally validated.")


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def show_image(path: Path, caption: str | None = None):
    path = Path(path)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Plot not available: `{rel(path)}` was not found.")


@st.cache_data(show_spinner=False)
def read_csv(path: str, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


def table_or_info(path: Path, **kw):
    df = read_csv(str(path), **kw)
    if df is None:
        st.info(f"Table not available: `{rel(path)}` was not found.")
    return df


@st.cache_data(show_spinner=False)
def read_text(path: str):
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_resource(show_spinner="Loading model ...")
def load_pipeline(model: str):
    return inf.load_pipeline(model)


@st.cache_data(show_spinner=False)
def feature_genes():
    return inf.load_feature_genes()


@st.cache_data(show_spinner=False)
def used_genes(model: str):
    return inf.load_used_genes(model)


@st.cache_data(show_spinner=False)
def gene_stats():
    return inf.load_gene_stats()


@st.cache_data(show_spinner=False)
def lr_background_mean():
    return np.load(inf.MODELS_DIR / "lr_shap_background_mean.npy")


def subtype_counts():
    """Subtype counts from the (small) split manifest; falls back to the documented numbers."""
    m = read_csv(str(ROOT / "data" / "processed" / "ml" / "ml_split_manifest.csv"))
    if m is not None and "subtype" in m:
        return m["subtype"].value_counts().reindex(CLASS_ORDER)
    return pd.Series({"Basal": 339, "Her2": 327, "LumA": 1657, "LumB": 729, "Normal": 221})
