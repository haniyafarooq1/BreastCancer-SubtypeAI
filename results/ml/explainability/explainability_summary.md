# Phase 5: Explainability summary

Script: `ml/04_explain.py`. Models were loaded from `models/` and not refit or changed.
All tables and plots are in this folder.

## How to read these results (caveats first)
- **Association, not causation.** Coefficients and SHAP values show how *this model* uses a gene
  to separate PAM50 labels. They do not show that a gene drives a subtype biologically.
- **Correlated genes share or swap importance.** Many genes are strongly co-expressed (e.g. the
  ER/luminal module, the proliferation module, the basal keratins). The L2-regularised LR spreads
  weight across such modules, and RF picks one gene from a module somewhat arbitrarily. A gene
  that ranks low may simply be redundant with a higher-ranked partner.
- **LR's Basal signal is spread out.** In the Basal beeswarm, no single gene stands out: nearly
  all of the Basal log-odds comes from the sum of many small contributions. The same pattern
  appears in the waterfall plots. Per-gene LR ranks are therefore most informative for the
  LumA / LumB / Her2 / Normal distinctions and least informative for Basal.
- **Scales differ.** LR SHAP is on the log-odds scale (LinearExplainer, interventional;
  explained = all 655 test samples; additivity error 5e-14). Background: a random sample of
  300 transformed training samples was passed in, but shap's default masker keeps only 100 of
  them, so the effective baseline is the mean of 100 training samples. RF SHAP is on the probability scale (TreeExplainer, stratified 200 test samples).
  Compare ranks, not raw values.
- **LR directions are relative.** In a multinomial model, "higher expression pushes towards
  subtype X" means relative to the other subtypes.
- **These genes are exploratory.** They come from one cohort (SCAN-B) and are not clinically
  validated biomarkers.
- **The labels are expression-derived.** PAM50 labels are themselves computed from expression of
  the PAM50 genes, so recovering PAM50 genes and getting high accuracy are partly expected (see
  below).

## 1. Logistic Regression (primary model)
Top 20 genes by global mean |SHAP| (averaged over subtypes): NAT1, SLC39A6, ERBB2, CDH3, MAPT,
NUF2, BPI, GRB7, CRIP1, PKIB, PGR, TSPAN1, TMEM45B, ESR1, SEMA6D, ATP13A5, GRIA4, VIM-AS1,
ST8SIA6-AS1, ZIC1 (`lr_top20_genes.csv`, `lr_shap_global_top20.png`).

Top genes per subtype by class-specific mean |SHAP|. ↑ = higher expression pushes towards the
subtype, ↓ = away (`lr_top_genes_per_subtype.csv`):

| Subtype | Top 10 genes |
|---|---|
| Basal | MSX2↓, BAIAP2L2↑, NDNF↓, IL22RA2↑, SLC9A3R1↓, SERPINA3↑, BCAS1↓, ERBB2↓, GJB3↑, PRRT3↓ (diffuse; see caveat) |
| Her2 | ERBB2↑, SLC39A6↓, ESR1↓, TMEM45B↑, GRB7↑, NAT1↓, CDH3↑, SPATA18↓, TSPAN1↑, CHODL↓ |
| LumA | NAT1↑, MAPT↑, NUF2↓, PGR↑, LOC101054525↑, BCL2↑, SEMA6D↓, KCNK15↑, PTTG1↓, BPI↓ |
| LumB | CDH3↓, ESR1↑, SLC39A6↑, KCNE1↓, NAT1↑, NUF2↑, FAM189A2↓, MAPT-IT1↓, CRIP1↑, MIA-RAB4B↓ |
| Normal | NAT1↓, MMP1↓, LINC00162↑, RIMS4↓, IGF1R↑, TAC1↑, NTF4↑, SYTL2↓, BPI↑, UBE2T↓ |

These largely follow the known biology. Her2 is driven by the 17q12 amplicon (ERBB2, GRB7) and
low ER signalling (ESR1↓, SLC39A6↓). LumA by ER/PR and BCL2 up and proliferation down (NUF2,
PTTG1, BIRC5, CDC6). LumB by ER up with proliferation up (NUF2↑) relative to LumA.

The top 15 positive and top 15 negative coefficients per subtype are in
`lr_top_coefficients_per_subtype.csv`; all 1,000 × 5 coefficients are in
`lr_coefficients_all.csv`. Plots: `lr_shap_beeswarm_<subtype>.png` (colour = log2 expression).

Waterfall examples (`lr_waterfall_examples.csv`):
- `lr_waterfall_basal_correct_Basal.png`: sample F2815, true Basal, P(Basal) ≈ 1.00.
- `lr_waterfall_normal_misclassified_as_luma_{LumA,Normal}.png`: sample F2606, true Normal,
  predicted LumA with P = 0.996. Luminal genes (MAPT, PGR) push it towards LumA. Low NAT1
  pushes it away from LumA, but most of the push comes from many small contributions.

## 2. Random Forest
Top 20 genes by SHAP: ESR1, RERG, FOXA1, TBC1D9, SFRP1, MIA, SLC39A6, KRT5, GATA3, BCL2, THSD4,
KRT14, PTTG1, NAT1, MLPH, BC070495, SPDEF, FOXC1, IL6ST, AURKA. Impurity importance (MDI) gives a
very similar list (`rf_importance.csv`, `rf_top20_genes.png`). Unlike LR, RF concentrates its
importance on classic luminal/basal markers. Note that MDI is biased towards high-variance
features.

## 3. SVC
KernelExplainer was skipped (too slow). Permutation importance was **skipped**: one test-set
scoring of the calibrated SVC takes 0.70 s, so 500 genes × 5 repeats was estimated at
≈29 min, over the 10-minute budget.

## 4. Comparison with the statistical analysis
The differential-expression results were computed on the full dataset (train + test), so they are
used here for comparison only. For the **statistical rank**, each gene is ranked by its maximum
|rank-biserial effect size| over the 10 pairwise comparisons in `results/biomarker_candidates/`.
The Kruskal-Wallis table
cannot be ranked by p-value because all adjusted p-values are 0.0, so the H statistic is used
as a check.

**20 biomarker candidates from the differential-expression analysis** (`biomarker_candidates_vs_ml.csv`):
- All 20 are in the training top-5000 variance genes and all 20 are selected by LR's SelectKBest.
- 17 of 20 are selected by RF's SelectKBest. ABCC11, MYBL2 and RRM2 are not; RF uses k=500.
- 4 are in the LR top 50: ESR1 (#14), BIRC5 (#34), PTTG1 (#36), BCL2 (#45).
- 15 are in the RF top 50.

**Top-50 overlap** (random expectation ≈ 0.09 genes, out of 28,604 tested):

| Comparison | Overlap |
|---|---|
| LR top 50 ∩ statistical top 50 (max effect) | **5** (ESR1, KRT17, MAPT, MGC4836, TBC1D9) |
| RF top 50 ∩ statistical top 50 (max effect) | **25** |
| LR top 50 ∩ statistical top 50 (Kruskal H) | 4 (BIRC5, ESR1, PTTG1, TBC1D9) |
| LR top 50 ∩ RF top 50 | 9 |

Why the LR overlap is low: the statistical top 50 is dominated by the most extreme pairwise contrasts,
Basal vs LumA (25 genes) and LumB vs Normal (22 genes), with |effect| between 0.96 and 0.99.
Those classes are already easy for the model. The LR top 50 is mostly about the *hard*
boundaries: by most-affected subtype it splits 19 LumA, 17 LumB, 8 Her2, 5 Normal and only 1
Basal.

**"New" genes:** 45 genes are in the LR top 50 but not in the statistical top 50
(`new_ml_genes_not_in_statistical_top50.csv`). None are new in a statistical sense: all 45 are
significant in the Kruskal-Wallis test. They are ranked lower by single-gene effect size
(median rank 961). Examples include ERBB2 (statistical rank 1,764) and GRB7 (7,107): they mark only
the Her2 subtype, so their maximum pairwise effect is moderate, but they are essential to the
multi-class model. 12 of the 45 are PAM50 genes. Genes that are neither PAM50 nor in the statistical top 50
(e.g. BPI, CRIP1, PKIB, TSPAN1, SEMA6D, MMP1, IL17B, DSG3) are exploratory leads only.

## 5. PAM50 check (Parker et al., J Clin Oncol 2009)
All 50 PAM50 genes are present under their current names (ORC6, NDC80, NUF2; no aliases needed).

| | Count |
|---|---|
| PAM50 genes in dataset | 50 |
| In training top-5000 variance | 41 (missing: ACTR3B, BAG1, BLVRA, CCNB1, CXXC5, GPR160, MDM2, MYC, TYMS) |
| Selected by LR (k=1000) | 39 |
| In LR top 50 | **15** (BCL2, BIRC5, CDH3, ERBB2, ESR1, GRB7, KRT17, MAPT, NAT1, NUF2, PGR, PTTG1, SLC39A6, TMEM45B, UBE2T) |
| Selected by RF (k=500) | 38 |
| In RF top 50 | **16** (BCL2, BIRC5, ERBB2, ESR1, FOXA1, FOXC1, KRT14, KRT17, KRT5, MIA, MLPH, NAT1, PTTG1, SFRP1, SLC39A6, UBE2C) |
| In statistical top 50 | 11 |

If 50 genes were drawn at random from the 1,000 LR features, about 2 would be PAM50 genes. The
observed 15 is about 7× that. The 9 PAM50 genes removed by the variance pre-filter vary little
in this cohort; their information is likely carried by correlated genes.

**Limitation for the report:** the SCAN-B PAM50 labels were assigned from the expression of
these same genes. Finding PAM50 genes among the top features, and the high accuracy itself, is
therefore partly built in. The models learn to reproduce an expression-based classifier; they
are not predicting an independent clinical outcome.
