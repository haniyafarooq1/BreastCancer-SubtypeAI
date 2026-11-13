"""Central configuration for the ML pipeline (paths, labels, CV settings)."""
import os
from pathlib import Path

# Repository root (this file lives in <root>/ml/config.py)
ROOT = Path(__file__).resolve().parents[1]

# ---- Input data (outputs of the existing preprocessing pipeline; read-only) ----
PRIMARY_CSV = ROOT / "data" / "processed" / "primary_dataset.csv"
SPLIT_MANIFEST = ROOT / "data" / "processed" / "ml" / "ml_split_manifest.csv"
EXISTING_RESULTS = ROOT / "results"

# ---- Cache folder (parquet copy of the dataset + pipeline cache during training) ----
# Order: BC_CACHE_DIR env var -> C:\bc_cache (Windows) or ~/.cache/bcsubtype (macOS/Linux)
# -> <repo>/.cache (gitignored) if the default cannot be created. Keeping the cache outside
# synced folders (OneDrive/Dropbox) avoids syncing ~400 MB.
def _resolve_cache_dir():
    candidates = []
    if os.environ.get("BC_CACHE_DIR"):
        candidates.append(Path(os.environ["BC_CACHE_DIR"]))
    candidates.append(Path(r"C:\bc_cache") if os.name == "nt" else Path.home() / ".cache" / "bcsubtype")
    candidates.append(ROOT / ".cache")
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("ok"); probe.unlink()
            return d
        except OSError:
            continue
    raise OSError(f"No writable cache folder among {candidates}")


CACHE_DIR = _resolve_cache_dir()
CACHE_PATH = Path(os.environ.get("BC_CACHE_PATH", CACHE_DIR / "primary_dataset.parquet"))
PIPELINE_CACHE = Path(os.environ.get("BC_PIPELINE_CACHE", CACHE_DIR / "pipeline_cache"))

# ---- Our outputs ----
MODELS_DIR = ROOT / "models"
ML_RESULTS = ROOT / "results" / "ml"
CV_DIR = ML_RESULTS / "cv"
EVAL_DIR = ML_RESULTS / "evaluation"
EXPLAIN_DIR = ML_RESULTS / "explainability"
APP_DIR = ROOT / "app"

# ---- Labels ----
TARGET = "subtype"
ID_COL = "sample_id"
CLASS_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]  # alphabetical = sklearn order

# Columns that must never be used as features
NON_FEATURE_COLS = {"subtype", "sample_id", "geo_accession", "split", "title"}

# ---- Reproducibility / CV ----
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_macro"

# The 20 exploratory biomarker candidates from the differential-expression analysis (results/analysis_summary.md, section 6).
# Computed on the FULL dataset -> comparison only, never used for feature selection.
DE_BIOMARKER_CANDIDATES = [
    "FOXC1", "FOXA1", "ESR1", "GATA3", "MLPH", "SPDEF", "AR", "ABCC11", "SOX10", "KRT5",
    "KRT14", "FOXM1", "UBE2C", "BIRC5", "AURKA", "PTTG1", "MYBL2", "RRM2", "BCL2", "XBP1",
]
