# Model selection rule

**Written:** 2026-11-13 19:19 PST, after all hyper-parameter tuning (Phase 3 + 3b grid extension)
was finished and **before** any model was evaluated on the test set.

## Rule

> The primary model is the one with the highest 5-fold CV macro F1 on the training set.
> If the top two are within one CV standard deviation of each other, choose Logistic
> Regression (more interpretable, better Normal-class recall and higher balanced accuracy).
> Test-set results are reported for all three models but are NOT used to choose the model.

## Application (from `results/ml/cv/cv_summary.csv`)

| Rank | Model | CV macro F1 (mean ± std) | CV balanced accuracy |
|---|---|---|---|
| 1 | SVC (RBF, C=100, k=500) | 0.8844 ± 0.0049 | 0.8760 ± 0.0047 |
| 2 | Logistic Regression (C=0.1, k=1000) | 0.8819 ± 0.0100 | 0.8890 ± 0.0105 |
| 3 | Random Forest (300 trees, min_samples_leaf=3, k=500) | 0.8536 ± 0.0162 | 0.8336 ± 0.0191 |

Difference between the top two: 0.8844 − 0.8819 = **0.0025**, which is smaller than
either model's CV standard deviation (0.0049 for SVC, 0.0100 for LR).
The top two are therefore within one CV standard deviation.

**Selected primary model: Logistic Regression** (`models/logreg_pipeline.joblib`).
