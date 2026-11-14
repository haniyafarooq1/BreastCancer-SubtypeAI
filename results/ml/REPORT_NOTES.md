# Report notes: machine learning & explainability (copy-ready)

## Key numbers
- **Data:** GSE96058 (SCAN-B), 3,273 tumours × 30,865 genes, 5 PAM50 subtypes (LumA 1,657 ·
  LumB 729 · Basal 339 · Her2 327 · Normal 221). Stratified 80/20 split: 2,618 train / 655 test.
- **Pipeline (leakage-safe, re-fit in every CV fold):** top-5,000 variance genes → StandardScaler →
  SelectKBest (ANOVA F, k tuned) → classifier with balanced class weights. 5-fold stratified CV,
  scored by macro F1.
- **Model selection** (rule fixed before testing): highest CV macro F1, with Logistic Regression
  preferred when the top two are within one CV SD. SVC 0.884 ± 0.005 vs LR 0.882 ± 0.010 →
  **Logistic Regression** (C = 0.1, 1,000 genes).
- **Test performance of the primary model (LR):** macro F1 **0.891**, accuracy **0.919**, balanced
  accuracy **0.894**. SVC: macro F1 0.897 / accuracy 0.921. RF: 0.832 / 0.879.
  The CV-to-test change is +0.009 for LR, so there is no sign of overfitting.
- **Per-class test F1 (LR):** Basal 0.93 · Her2 0.91 · LumA 0.94 · LumB 0.92 · **Normal 0.76** (weakest).
- **Main errors:** Normal → LumA (7/44, 16%), LumB → LumA (6%), LumA → LumB (3%).
- **Explainability:** LR top genes include NAT1, SLC39A6, ERBB2, CDH3, MAPT, GRB7, PGR, ESR1.
  The RF top genes are classic markers (ESR1, FOXA1, SFRP1, KRT5, KRT14, FOXC1, GATA3).
  **15 of the LR top 50 genes are PAM50 genes** (≈2 expected by chance).
  Overlap with the statistical top 50 is 5/50 for LR and 25/50 for RF (≈0.09 by chance).
  The LR genes target the hard LumA/LumB/Her2/Normal boundaries; the statistical ranking is
  dominated by the easy Basal vs luminal contrast.
- **Interactive app:** Streamlit, 5 pages. Predicts from uploaded expression data; only the 500–1,000
  genes a model uses are needed, verified to give identical predictions to the full input.

## Suggested figures
1. **Confusion matrix of the primary model (counts + recall):**
   `results/ml/evaluation/confusion_matrix_logreg.png`.
   Caption idea: *"Test-set confusion matrix of the Logistic Regression model (n = 655). Most errors
   are Normal → LumA and LumA ↔ LumB."*
2. **Global gene importance (LR SHAP, top 20, split by subtype):**
   `results/ml/explainability/lr_shap_global_top20.png`
3. **Per-subtype explanation, e.g. Her2 beeswarm** (shows ERBB2/GRB7 up, ESR1 down):
   `results/ml/explainability/lr_shap_beeswarm_Her2.png`
4. **Random Forest marker genes (SHAP vs impurity importance):**
   `results/ml/explainability/rf_top20_genes.png`.
   Useful next to figure 2 for the Basal caveat.

Optional: a model comparison table (from `results/ml/evaluation/model_comparison.csv`) or
`results/ml/explainability/lr_waterfall_normal_misclassified_as_luma_LumA.png` as a worked
misclassification example.

## Limitations (for the discussion section)
- Single cohort (SCAN-B) with no external validation; other platforms or normalisations may perform worse.
- Circularity: the PAM50 labels are themselves derived from gene expression (largely the PAM50
  genes), so high accuracy and the recovery of PAM50 genes are partly built in.
- Basal-like (PAM50) is not the same as triple-negative breast cancer (clinical ER/PR/HER2 status).
- Normal-like is the weakest class and has only 44 test samples, so its metrics are noisy
  (one sample ≈ 2.3 points of recall).
- SHAP values and coefficients show model associations, not causation. Correlated genes share
  importance; L2-regularised LR spreads the Basal signal across many genes.
- The candidate genes are exploratory and not clinically validated. The app is a research and
  educational tool, not for clinical diagnosis.
