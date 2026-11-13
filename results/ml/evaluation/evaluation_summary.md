# Phase 4: test-set evaluation

Generated 2026-11-13 19:22. Test set: 655 samples, evaluated once. Primary model (chosen by CV rule before testing): **Logistic Regression**.

## Test metrics
| model_name | test_accuracy | test_balanced_accuracy | test_precision_macro | test_recall_macro | test_f1_macro | test_f1_weighted | test_log_loss |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.919 | 0.894 | 0.888 | 0.894 | 0.891 | 0.919 | 0.259 |
| SVC (RBF, calibrated) | 0.921 | 0.881 | 0.918 | 0.881 | 0.897 | 0.919 | 0.298 |
| Random Forest | 0.879 | 0.817 | 0.852 | 0.817 | 0.832 | 0.877 | 0.354 |

## Overfitting check (macro F1)
| model_name | train_fit_f1_macro | cv_f1_macro_mean | cv_f1_macro_std | test_f1_macro | test_minus_cv_f1_macro |
|---|---|---|---|---|---|
| Logistic Regression | 0.999 | 0.882 | 0.010 | 0.891 | 0.009 |
| SVC (RBF, calibrated) | 1.000 | 0.884 | 0.005 | 0.897 | 0.012 |
| Random Forest | 0.994 | 0.854 | 0.016 | 0.832 | -0.021 |

## Per-class F1: CV vs test
| model | CV Basal | CV Her2 | CV LumA | CV LumB | CV Normal | Test Basal | Test Her2 | Test LumA | Test LumB | Test Normal |
|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.957 | 0.870 | 0.931 | 0.873 | 0.780 | 0.933 | 0.908 | 0.941 | 0.919 | 0.756 |
| svc | 0.952 | 0.880 | 0.936 | 0.882 | 0.776 | 0.943 | 0.898 | 0.941 | 0.907 | 0.795 |
| rf | 0.942 | 0.848 | 0.923 | 0.869 | 0.688 | 0.925 | 0.866 | 0.910 | 0.871 | 0.590 |

## Main confusions on the test set (true -> predicted, share of the true class)
| true -> predicted | logreg | svc | rf |
|---|---|---|---|
| Normal -> LumA | 7/44 (16%) | 9/44 (20%) | 18/44 (41%) |
| LumA -> Normal | 7/332 (2%) | 2/332 (1%) | 6/332 (2%) |
| LumA -> LumB | 11/332 (3%) | 9/332 (3%) | 14/332 (4%) |
| LumB -> LumA | 9/146 (6%) | 14/146 (10%) | 17/146 (12%) |
| Her2 -> LumB | 2/65 (3%) | 3/65 (5%) | 5/65 (8%) |
| LumB -> Her2 | 1/146 (1%) | 1/146 (1%) | 1/146 (1%) |
## Notes

**Model choice.** Logistic Regression is the primary model, chosen by the CV rule written
before testing (`results/ml/model_selection_rule.md`). SVC has a slightly higher test macro F1
(0.897 vs 0.891), but the 0.005 gap means about 1–2 samples per class on a 655-sample
test set, so it does not change the choice. Per the rule, test results were not used to select.

**Overfitting check.** For LR and SVC the test macro F1 is within about 1 point of the CV
estimate (+0.009 and +0.012), so the CV estimate generalised. Train-fit F1 is ≈1.0 for all
models, which is expected with ~2.6k samples and 500–1000 genes; CV/test is the relevant
measure. RF is 2 points below CV and is the weakest model overall.

**Main confusions (all models).**
- **Normal → LumA** is the dominant error: 16% (LR), 20% (SVC), 41% (RF) of the 44 Normal
  test samples. Normal is the hardest class (test F1 0.59–0.80). "Normal-like" PAM50 calls
  partly reflect normal tissue content in the sample, and their luminal-like profile overlaps
  with LumA. The small test count (44) also makes per-class Normal metrics noisy (±1 sample
  ≈ ±2.3 points of recall).
- **LumA ↔ LumB**: 3–4% of LumA are predicted as LumB, and 6–12% of LumB as LumA. Both are
  luminal (ER+) subtypes, separated mainly by proliferation and lying on a continuum, so some
  boundary errors are expected.
- **Her2 → LumB** is small on the test set (2–5 of 65, 3–8%) but was clearly visible in CV
  (23–34 of 262), so it is a real but moderate confusion (the luminal-HER2 overlap).
- **Basal** is the most separable class (F1 0.93–0.94), with very few errors in either direction.
