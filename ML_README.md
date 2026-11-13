# BreastCancer-SubtypeAI: Machine learning, explainability & app

This document covers the machine-learning part of the project: subtype classifiers, explainability,
and the Streamlit app. Data preparation, EDA, differential expression and the train/test split are
in `notebooks/01`–`15` and `results/` (Haniya). Those files are used as input here and are not modified.

> ⚠️ **Research/educational tool. Not for clinical diagnosis.**

---

## 1. Setup

Tested with **Python 3.11** on Windows 11 (PowerShell). The steps below work on a fresh clone:

```powershell
# 1. clone and enter the repo
git clone <repo-url> BreastCancer-SubtypeAI
cd BreastCancer-SubtypeAI

# 2. create a virtual environment. Keep it OUTSIDE OneDrive/Dropbox, e.g. C:\venvs\bcsubtype
python -m venv C:\venvs\bcsubtype
C:\venvs\bcsubtype\Scripts\Activate.ps1          # macOS/Linux: source <venv>/bin/activate
#   if PowerShell blocks the script: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. install the pinned packages (takes about 5-8 min)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins exact versions. **scikit-learn must be 1.6.1**: the saved models were
trained with it, and pickled models are only guaranteed to load and behave identically with the
same version. The app shows a red warning in the sidebar if the installed version differs.

### Running the app only (no dataset needed)
The trained models, pre-computed results and a demo input file are in the repository, so the app
runs straight after setup:

```powershell
streamlit run app/app.py                 # opens http://localhost:8501
python app/tests/test_app_smoke.py       # automated smoke test, about 30 s, prints ALL CHECKS PASSED
```

### Re-running the ML pipeline (dataset needed)
The scripts need `data/processed/primary_dataset.csv` (about 1.8 GB, gitignored). Either copy it
from a teammate into that folder, or regenerate it with Haniya's scripts `notebooks/01`–`08`, which
need the raw GEO file in `data/raw/`. The split manifest `data/processed/ml/ml_split_manifest.csv`
is in the repository.

**Cache folder.** The first run writes a float32 parquet copy of the dataset (about 400 MB). It
goes to `C:\bc_cache` on Windows or `~/.cache/bcsubtype` on macOS/Linux. If that folder can't be
created, it falls back to `<repo>/.cache/` (gitignored). To choose another location, set
`BC_CACHE_DIR` (PowerShell: `$env:BC_CACHE_DIR = "D:\bc_cache"`). Keep it out of synced folders.

**Memory.** Everything runs single-process (`n_jobs=1`) and fits on an 8 GB laptop. Close
memory-heavy programs (e.g. the browser) during training.

---

## 2. Scripts and execution order

Run from the repository root with the venv active.

| # | Script | What it does | Produces | Time* |
|---|---|---|---|---|
| 1 | `python ml/01_load_split.py` | Loads the dataset as float32, builds the parquet cache, applies the existing split manifest, asserts no train/test overlap, no `subtype` among features, no `repl` IDs | parquet cache; printed shapes/counts | ~1 min first run, ~5 s after |
| 2 | `python ml/02a_smoke_test.py` | *(optional)* unit checks for the variance filter, quick 3-fold LR, memory/timing test | printed report | ~1.5 min |
| 3 | `python ml/02_train_cv.py` | 5-fold GridSearchCV for LR / SVC / RF on the **training set only**; per-class CV F1; refit on the full training set; save models | `results/ml/cv/*`, `models/*_pipeline.joblib`, `models/*.json` | ~12 min |
| 4 | `python ml/02b_grid_extension.py` | Grid extensions for best values on the grid edge (SVC C=100, k=250; RF k=250); re-saves a model only if its best params change | updated `results/ml/cv/*`, `models/svc_pipeline.joblib` | 3–15 min |
| 5 | `python ml/03_evaluate.py` | **One-time** test-set evaluation of the 3 saved models (no fitting). Refuses to run again unless `--force` | `results/ml/evaluation/*`, `models/best_model.json` | ~30 s |
| 6 | `python ml/04_explain.py` | LR coefficients + SHAP, RF SHAP + MDI, comparison with Haniya's biomarkers, PAM50 check | `results/ml/explainability/*` | ~30 s |
| 7 | `python ml/05_prepare_app_assets.py` | App assets (training-set gene medians/percentiles, SHAP baseline, demo file); verifies that "only used genes" input gives the same predictions as the full input | `models/train_gene_stats.csv.gz`, `models/lr_shap_background_mean.npy`, `app/sample_input/demo_samples.csv`, `results/ml/app_checks/*` | ~1 min |

\*Measured on the development laptop (8 GB RAM). The model-selection rule
(`results/ml/model_selection_rule.md`) was written after step 4 and before step 5.

Supporting modules: `ml/config.py` (paths, labels, seeds), `ml/data.py` (loading + split checks),
`ml/transformers.py` (`TopVarianceSelector`), `ml/pipeline.py` (pipeline factory),
`ml/training.py` (shared training helpers), `ml/inference.py` (shared by the app and step 7).

---

## 3. Methods

**Data.** GSE96058 (SCAN-B): 3,273 primary tumours (technical replicates removed), 30,865 genes,
log2(FPKM + 0.1). Target: 5 PAM50 subtypes, Basal 339 · Her2 327 · LumA 1,657 · LumB 729 · Normal 221.

**Split.** The existing stratified 80/20 split (`random_state=42`) is reused from
`ml_split_manifest.csv`, matched by sample ID: 2,618 training and 655 test samples. Per-class counts
were verified after matching.

**Leakage-safe pipeline.** All preprocessing is inside one scikit-learn `Pipeline`, so it is re-fit
on the training part of every CV fold:
1. `TopVarianceSelector`: keep the 5,000 most variable genes (variance computed on the fit data only)
2. `StandardScaler`
3. `SelectKBest(f_classif, k)` with k tuned in {250, 500, 1000, 2000}
4. Classifier with `class_weight="balanced"`

The test set was not used for feature selection, scaling or tuning. Haniya's biomarker results
(computed on the full dataset) were used only for comparison, never for feature selection.

**Tuning.** `StratifiedKFold(5, shuffle=True, random_state=42)`, `GridSearchCV` with
`scoring = {f1_macro, balanced_accuracy, accuracy}` and `refit="f1_macro"`.
- Logistic Regression (lbfgs, L2): C ∈ {0.001, 0.01, 0.1, 1}
- SVC: linear kernel with C ∈ {0.001, 0.01, 0.1}; RBF kernel with C ∈ {1, 10, 100}, gamma="scale"
- Random Forest (300 trees, max_features="sqrt"): min_samples_leaf ∈ {1, 3}

When a best value landed on the edge of its grid, the grid was extended once (step 4).

**Model-selection rule** (written down before any test-set evaluation): the primary model has the
highest CV macro F1. If the top two are within one CV standard deviation, Logistic Regression is
chosen (more interpretable, better Normal recall, higher balanced accuracy). Test results are
reported for all models but not used for selection.

**Calibration / probabilities.** The final SVC is wrapped in
`CalibratedClassifierCV(cv=5, method="sigmoid")` and refit on the full training set. For all three
models, `predict` equals `argmax(predict_proba)` (0 mismatches on 2,618 training samples).

---

## 4. Results

### Cross-validation (training set, 5 folds)
| Model | Best parameters | CV macro F1 | CV balanced accuracy |
|---|---|---|---|
| SVC | RBF kernel, C=100, k=500 | **0.884 ± 0.005** | 0.876 ± 0.005 |
| Logistic Regression | C=0.1, k=1000 | 0.882 ± 0.010 | **0.889 ± 0.010** |
| Random Forest | min_samples_leaf=3, k=500 | 0.854 ± 0.016 | 0.834 ± 0.019 |

SVC and LR differ by 0.0025, less than one CV standard deviation, so the rule selects
**Logistic Regression** as the primary model. Remaining grid-edge note: SVC C=100 is at the upper
edge of its grid; it was not extended further (+0.001 over C=10).

### Test set (655 samples, evaluated once)
| Model | Accuracy | Balanced acc. | Macro F1 | Weighted F1 | CV → test macro F1 |
|---|---|---|---|---|---|
| **Logistic Regression (primary)** | 0.919 | **0.894** | 0.891 | 0.919 | 0.882 → 0.891 |
| SVC (calibrated) | **0.921** | 0.881 | **0.897** | 0.919 | 0.884 → 0.897 |
| Random Forest | 0.879 | 0.817 | 0.832 | 0.877 | 0.854 → 0.832 |

Test scores for LR and SVC are within about 1 point of CV, so there is no sign of overfitting.
SVC's 0.005 higher test macro F1 (about 1–2 samples per class) does not change the pre-registered choice.

### Per-class F1 (CV → test)
| Model | Basal | Her2 | LumA | LumB | Normal |
|---|---|---|---|---|---|
| LR | 0.957 → 0.933 | 0.870 → 0.908 | 0.931 → 0.941 | 0.873 → 0.919 | 0.780 → 0.756 |
| SVC | 0.952 → 0.943 | 0.880 → 0.898 | 0.936 → 0.941 | 0.882 → 0.907 | 0.776 → 0.795 |
| RF | 0.942 → 0.925 | 0.848 → 0.866 | 0.923 → 0.910 | 0.869 → 0.871 | 0.688 → 0.590 |

### Main confusions (test set, share of the true class)
| True → predicted | LR | SVC | RF |
|---|---|---|---|
| Normal → LumA | 7/44 (16%) | 9/44 (20%) | 18/44 (41%) |
| LumB → LumA | 9/146 (6%) | 14/146 (10%) | 17/146 (12%) |
| LumA → LumB | 11/332 (3%) | 9/332 (3%) | 14/332 (4%) |
| Her2 → LumB | 2/65 (3%) | 3/65 (5%) | 5/65 (8%) |

- **Normal → LumA** is the dominant error.
- **LumA ↔ LumB** are luminal subtypes on a proliferation continuum.
- **Her2 → LumB** was clearer in CV (23–34 of 262).
- **Basal** is the most separable subtype.

Details: `results/ml/evaluation/evaluation_summary.md`.

---

## 5. Explainability

SHAP (LinearExplainer for LR on all 655 test samples, log-odds scale; TreeExplainer for RF on 200
stratified test samples) and LR coefficients. Full write-up: `results/ml/explainability/explainability_summary.md`.

**LR top genes per subtype** (↑ = higher expression pushes towards the subtype, ↓ = away):
| Subtype | Top genes |
|---|---|
| Her2 | ERBB2↑, SLC39A6↓, ESR1↓, TMEM45B↑, GRB7↑, NAT1↓, CDH3↑ |
| LumA | NAT1↑, MAPT↑, NUF2↓, PGR↑, BCL2↑, SEMA6D↓, PTTG1↓ |
| LumB | CDH3↓, ESR1↑, SLC39A6↑, KCNE1↓, NAT1↑, NUF2↑, CRIP1↑ |
| Normal | NAT1↓, MMP1↓, LINC00162↑, RIMS4↓, IGF1R↑, TAC1↑, NTF4↑ |
| Basal | diffuse (see caveat): MSX2↓, BAIAP2L2↑, NDNF↓, IL22RA2↑, … |

Global LR top 5: NAT1, SLC39A6, ERBB2, CDH3, MAPT. These match known biology: the ERBB2/GRB7
amplicon for Her2; ER/PR/BCL2 up and proliferation down for LumA; ER plus proliferation for LumB.

**Basal caveat.** The L2-regularised LR spreads its Basal weight across many correlated genes.
Almost all Basal signal is in the "sum of other features" term, so single-gene LR ranks for Basal
are not informative. The RF view shows the classic markers instead: ESR1, FOXA1, SFRP1, KRT5,
KRT14, FOXC1, MLPH, SPDEF.

**Comparison with Haniya's statistics.**
- All 20 biomarker candidates pass the variance filter and are selected by LR; 17/20 are selected by RF.
- 4 are in the LR top 50 (ESR1, BIRC5, PTTG1, BCL2) and 15 are in the RF top 50.
- Top-50 overlap with her effect-size ranking: **LR 5/50, RF 25/50** (≈0.09 expected by chance).
- LR overlaps little because her top 50 comes from the easiest contrasts (Basal vs LumA,
  LumB vs Normal). The LR top 50 targets the hard LumA/LumB/Her2/Normal boundaries.
- The 45 "new" LR genes are all significant in her Kruskal–Wallis test, just ranked lower.

**PAM50 check** (Parker et al., *J Clin Oncol* 2009):
- All 50 PAM50 genes are in the data; 41 pass the variance filter.
- 39 are in the LR model's 1,000 genes and 38 in RF's 500.
- **15 are in the LR top 50** (about 2 expected by chance) and 16 in the RF top 50.

---

## 6. Streamlit app (`app/`)

| Page | Content |
|---|---|
| Home (`app.py`) | Objective, dataset, pipeline, headline results, team, Basal ≠ TNBC note |
| 1 Dataset Analysis | Haniya's EDA plots (subtype distribution, expression distribution, PCA, heatmap) and DE tables |
| 2 Model Comparison | Test metrics, CV vs test, per-class F1, confusion matrices, selection rule |
| 3 Subtype Prediction | Upload a CSV or use demo samples; predicted subtype, probabilities, per-sample LR SHAP; CSV download |
| 4 Explainability | LR SHAP/coefficients, RF markers, Basal caveat, biomarker/PAM50 comparison |

**Prediction input.**
- **Format:** CSV with samples as rows and HGNC gene symbols as columns, plus an optional `sample_id` column. A file with genes as rows is detected and transposed.
- **Scale:** values must be log2(FPKM + 0.1).
- **Required genes:** only the genes the selected model uses (LR 1,000; SVC/RF 500). A template can be downloaded per model.
- **Other genes:** filled with the training-set median. They are removed inside the pipeline, so predictions are identical to a full 30,865-gene input (verified bitwise for the model features: `results/ml/app_checks/full_vs_used_genes_check.csv`).
- **Refusals and warnings:** if any required gene is missing or non-numeric, the app refuses to predict and lists the genes. It also warns about count-like or off-scale values.

The app loads only the saved models, JSON files and pre-computed results. It never loads the dataset.

---

## 7. Limitations

- **Single cohort, no external validation.** All results come from SCAN-B (one RNA-seq protocol and
  normalisation). Performance on other cohorts, platforms (microarray, other RNA-seq pipelines) or
  normalisations is unknown and probably lower.
- **Circularity of the labels.** The PAM50 labels were themselves derived from gene expression
  (largely the PAM50 genes). High accuracy and the recovery of PAM50 genes are therefore partly built
  in: the models reproduce an expression-based classifier rather than predicting an independent
  clinical or biological ground truth.
- **Basal-like ≠ TNBC.** Basal-like is a PAM50 expression subtype. Triple-negative breast cancer is
  defined clinically by ER/PR/HER2 status (IHC/ISH). They overlap but are not the same.
- **Normal-like is the weakest class** (test F1 0.59–0.80, mostly confused with LumA). It has only 44
  test samples, so its per-class metrics are noisy: one sample is about ±2.3 points of recall.
  Normal-like calls also partly reflect normal-tissue content in the sample.
- **Explanations are associations, not causation.** SHAP values and coefficients show what the
  models use. Correlated genes share or swap importance, and the genes are exploratory, not
  clinically validated biomarkers.
- **Not a clinical tool.** For research and education only; not for diagnosis or treatment decisions.
