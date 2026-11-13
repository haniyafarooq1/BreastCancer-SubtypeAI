"""Phase 5: explainability for the saved (final, unchanged) pipelines.

LR (primary): coefficients, SHAP LinearExplainer (background = transformed TRAIN sample,
              explained = full test set), global / per-class plots, waterfalls.
RF:           SHAP TreeExplainer on a stratified test subset + impurity importance.
SVC:          permutation importance on the test set only if estimated < ~10 min.
Comparison:   Haniya's 20 biomarker candidates, her top-50 statistical genes, PAM50.

Models are only loaded, never refit. Run from repo root:  python ml/04_explain.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

from ml import config as C
from ml.data import load_split

OUT = C.EXPLAIN_DIR
OUT.mkdir(parents=True, exist_ok=True)
CLS = C.CLASS_ORDER
SUBTYPE_COLORS = dict(zip(CLS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))  # validated
BLUE, INK, MUTED = "#2a78d6", "#1f1f1e", "#6b6b66"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec7f7e", "#a8282a"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})
RNG = np.random.default_rng(C.RANDOM_STATE)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def savefig(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stratified_sample(y, n, seed=C.RANDOM_STATE):
    frac = n / len(y)
    idx = (y.groupby(y, group_keys=False)
             .apply(lambda s: s.sample(max(1, int(round(len(s) * frac))), random_state=seed)).index)
    return list(idx)


# ------------------------------------------------------------------ data + models
data, feats = load_split(which=("train", "test"))
X_tr, y_tr = data["train"]
X_te, y_te = data["test"]
lr = joblib.load(C.MODELS_DIR / "logreg_pipeline.joblib")
rf = joblib.load(C.MODELS_DIR / "rf_pipeline.joblib")
svc = joblib.load(C.MODELS_DIR / "svc_pipeline.joblib")
log("data + models loaded")

var_genes_train = set(lr.named_steps["variance"].get_feature_names_out())  # same 5000 for all models
assert var_genes_train == set(rf.named_steps["variance"].get_feature_names_out())

# =================================================================== 1. LOGISTIC REGRESSION
lr_genes = np.array(lr[:-1].get_feature_names_out())
lr_model = lr.named_steps["model"]
Z_tr = lr[:-1].transform(X_tr).astype(np.float64)
Z_te = lr[:-1].transform(X_te).astype(np.float64)
raw_te = X_te[lr_genes].to_numpy(np.float64)  # log2 expression, for colouring / display

# ---- coefficients (scaled, selected features; one vector per subtype)
coef = pd.DataFrame(lr_model.coef_.T, index=lr_genes, columns=CLS)
coef.index.name = "gene"
coef.to_csv(OUT / "lr_coefficients_all.csv")
rows = []
for c in CLS:
    s = coef[c].sort_values()
    for rank, (g, v) in enumerate(s[::-1].head(15).items(), 1):
        rows.append({"subtype": c, "sign": "positive", "rank": rank, "gene": g, "coefficient": v,
                     "direction": f"higher expression pushes TOWARDS {c}"})
    for rank, (g, v) in enumerate(s.head(15).items(), 1):
        rows.append({"subtype": c, "sign": "negative", "rank": rank, "gene": g, "coefficient": v,
                     "direction": f"higher expression pushes AWAY from {c}"})
pd.DataFrame(rows).to_csv(OUT / "lr_top_coefficients_per_subtype.csv", index=False)
log("LR coefficients saved")

# ---- SHAP LinearExplainer
bg_idx = RNG.choice(len(Z_tr), size=300, replace=False)
expl = shap.LinearExplainer(lr_model, Z_tr[bg_idx])
sv = np.asarray(expl.shap_values(Z_te))          # (n_test, n_genes, n_classes), log-odds scale
base = np.asarray(expl.expected_value)
assert sv.shape == (len(Z_te), len(lr_genes), len(CLS)), sv.shape
# additivity check: base + sum(shap) == decision_function (per class, up to softmax-invariant shift)
dec = lr_model.decision_function(Z_te)
recon = base + sv.sum(axis=1)
add_err = np.abs((recon - recon.mean(1, keepdims=True)) - (dec - dec.mean(1, keepdims=True))).max()
log(f"LR SHAP computed {sv.shape}; additivity max error {add_err:.2e}")

per_class_abs = pd.DataFrame(np.abs(sv).mean(axis=0), index=lr_genes, columns=CLS)
lr_imp = pd.DataFrame({"mean_abs_shap": per_class_abs.mean(axis=1)})
lr_imp = lr_imp.join(per_class_abs.add_prefix("mean_abs_shap_"))
lr_imp["subtype_most_important"] = per_class_abs.idxmax(axis=1)
lr_imp["coef_in_that_subtype"] = [coef.loc[g, s] for g, s in lr_imp["subtype_most_important"].items()]
lr_imp["direction_in_that_subtype"] = np.where(lr_imp["coef_in_that_subtype"] > 0, "up", "down")
lr_imp = lr_imp.sort_values("mean_abs_shap", ascending=False)
lr_imp.insert(0, "rank", range(1, len(lr_imp) + 1))
lr_imp.index.name = "gene"
lr_imp.to_csv(OUT / "lr_shap_importance.csv")
lr_imp.head(20).to_csv(OUT / "lr_top20_genes.csv")

pcs = []
for c in CLS:
    s = per_class_abs[c].sort_values(ascending=False).head(15)
    for r, (g, v) in enumerate(s.items(), 1):
        pcs.append({"subtype": c, "rank": r, "gene": g, "mean_abs_shap": v, "coefficient": coef.loc[g, c],
                    "direction": "higher -> towards" if coef.loc[g, c] > 0 else "higher -> away"})
pcs = pd.DataFrame(pcs)
pcs.to_csv(OUT / "lr_top_genes_per_subtype.csv", index=False)

# ---- global plot: top 20, stacked by subtype contribution to the class-averaged mean |SHAP|
top20 = lr_imp.head(20).index[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
left = np.zeros(len(top20))
for c in CLS:
    vals = per_class_abs.loc[top20, c].to_numpy() / len(CLS)
    ax.barh(top20, vals, left=left, color=SUBTYPE_COLORS[c], label=c, height=0.72,
            edgecolor="white", linewidth=1)
    left += vals
ax.set_xlabel("mean |SHAP| (log-odds), averaged over subtypes")
ax.set_title("Logistic Regression: top 20 genes (test set, n=655)", loc="left", color=INK)
ax.legend(title="Subtype share", frameon=False, loc="lower right")
ax.grid(axis="x", color="#e6e5e1", linewidth=0.8); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
savefig(fig, "lr_shap_global_top20.png")

# ---- per-class beeswarms
for ci, c in enumerate(CLS):
    e = shap.Explanation(values=sv[:, :, ci], base_values=np.full(len(Z_te), base[ci]),
                         data=raw_te, feature_names=list(lr_genes))
    plt.figure()
    shap.plots.beeswarm(e, max_display=15, show=False, color=DIVERGING, plot_size=(8, 6.5))
    plt.title(f"LR SHAP: {c} (log-odds of {c}); colour = log2 expression", loc="left", fontsize=11)
    savefig(plt.gcf(), f"lr_shap_beeswarm_{c}.png")
log("LR plots saved")

# ---- waterfalls: one confident correct Basal, one Normal misclassified as LumA
proba = lr.predict_proba(X_te)
pred = lr.classes_[proba.argmax(1)]
ytv = y_te.to_numpy()
examples = []
m = np.flatnonzero((ytv == "Basal") & (pred == "Basal"))
examples.append(("basal_correct", m[proba[m, CLS.index("Basal")].argmax()], ["Basal"]))
m = np.flatnonzero((ytv == "Normal") & (pred == "LumA"))
if len(m):
    examples.append(("normal_misclassified_as_luma", m[proba[m, CLS.index("LumA")].argmax()], ["LumA", "Normal"]))
ex_rows = []
for tag, i, classes in examples:
    for c in classes:
        ci = CLS.index(c)
        e = shap.Explanation(values=sv[i, :, ci], base_values=base[ci], data=raw_te[i],
                             feature_names=list(lr_genes))
        plt.figure()
        shap.plots.waterfall(e, max_display=12, show=False)
        plt.title(f"Sample {X_te.index[i]} (true {ytv[i]}, predicted {pred[i]}, "
                  f"P({c})={proba[i, ci]:.2f}): explaining log-odds of {c}", fontsize=10, loc="left")
        savefig(plt.gcf(), f"lr_waterfall_{tag}_{c}.png")
    ex_rows.append({"example": tag, "sample_id": X_te.index[i], "true": ytv[i], "predicted": pred[i],
                    **{f"prob_{c}": proba[i, k] for k, c in enumerate(CLS)}})
pd.DataFrame(ex_rows).to_csv(OUT / "lr_waterfall_examples.csv", index=False)
log("LR waterfalls saved")

# =================================================================== 2. RANDOM FOREST
rf_genes = np.array(rf[:-1].get_feature_names_out())
rf_model = rf.named_steps["model"]
mdi = pd.Series(rf_model.feature_importances_, index=rf_genes).sort_values(ascending=False)

sub_ids = stratified_sample(y_te, 200)
Zr = rf[:-1].transform(X_te.loc[sub_ids]).astype(np.float64)
texpl = shap.TreeExplainer(rf_model)
t = time.time()
rsv = np.asarray(texpl.shap_values(Zr, check_additivity=False))  # (n, genes, classes), probability scale
log(f"RF TreeExplainer on {len(sub_ids)} test samples in {time.time() - t:.0f}s, shape {rsv.shape}")
rf_abs = pd.DataFrame(np.abs(rsv).mean(axis=0), index=rf_genes, columns=CLS)
rf_imp = pd.DataFrame({"mean_abs_shap": rf_abs.mean(axis=1), "mdi_importance": mdi})
rf_imp = rf_imp.join(rf_abs.add_prefix("mean_abs_shap_"))
rf_imp["subtype_most_important"] = rf_abs.idxmax(axis=1)
rf_imp = rf_imp.sort_values("mean_abs_shap", ascending=False)
rf_imp.insert(0, "shap_rank", range(1, len(rf_imp) + 1))
rf_imp.insert(1, "mdi_rank", rf_imp["mdi_importance"].rank(ascending=False, method="min").astype(int))
rf_imp.index.name = "gene"
rf_imp.to_csv(OUT / "rf_importance.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), constrained_layout=True)
for ax, col, ttl, xl in ((axes[0], "mean_abs_shap", "SHAP (TreeExplainer, 200 test samples)",
                          "mean |SHAP| (probability), averaged over subtypes"),
                         (axes[1], "mdi_importance", "Built-in impurity importance (MDI)",
                          "mean decrease in impurity")):
    s = rf_imp[col].sort_values(ascending=False).head(20)[::-1]
    ax.barh(s.index, s.values, color=BLUE, height=0.72)
    ax.set_title(ttl, loc="left", fontsize=11); ax.set_xlabel(xl)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
fig.suptitle("Random Forest: top 20 genes", x=0.01, ha="left", fontsize=12, color=INK)
savefig(fig, "rf_top20_genes.png")
log("RF importance saved")

# =================================================================== 3. SVC permutation importance
svc_genes = np.array(svc[:-1].get_feature_names_out())
Zs = svc[:-1].transform(X_te)
svc_model = svc.named_steps["model"]
N_REPEATS = 5
t = time.time()
from sklearn.metrics import f1_score
for _ in range(3):
    f1_score(y_te, svc_model.predict(Zs), average="macro")
t_one = (time.time() - t) / 3
est = t_one * len(svc_genes) * N_REPEATS
log(f"SVC: one scoring {t_one:.2f}s -> permutation importance estimate {est / 60:.1f} min")
svc_status = ""
if est < 600:
    t = time.time()
    pi = permutation_importance(svc_model, Zs, y_te, scoring="f1_macro", n_repeats=N_REPEATS,
                                random_state=C.RANDOM_STATE, n_jobs=1)
    svc_imp = pd.DataFrame({"importance_mean": pi.importances_mean, "importance_std": pi.importances_std},
                           index=svc_genes).sort_values("importance_mean", ascending=False)
    svc_imp.insert(0, "rank", range(1, len(svc_imp) + 1)); svc_imp.index.name = "gene"
    svc_imp.to_csv(OUT / "svc_permutation_importance_test.csv")
    svc_status = f"computed on the test set ({N_REPEATS} repeats, macro F1 drop) in {(time.time() - t) / 60:.1f} min"
else:
    svc_imp = None
    svc_status = f"skipped: estimated {est / 60:.0f} min (> 10 min budget)"
log(f"SVC permutation importance {svc_status}")

# =================================================================== 4. Haniya comparison
bc = []
for f in sorted((C.HANIYA_RESULTS / "biomarker_candidates").glob("*_vs_*.csv")):
    d = pd.read_csv(f, usecols=["gene", "effect_size", "adjusted_p_value"])
    d["comparison"] = f.stem
    bc.append(d)
bc = pd.concat(bc)
bc["abs_effect"] = bc["effect_size"].abs()
bc = bc.reset_index(drop=True)
best_pair = bc.loc[bc.groupby("gene")["abs_effect"].idxmax()].set_index("gene")
stat = best_pair[["abs_effect", "comparison", "effect_size"]].rename(
    columns={"abs_effect": "max_abs_effect", "comparison": "max_effect_comparison"})
stat = stat.sort_values("max_abs_effect", ascending=False)
stat.insert(0, "stat_effect_rank", range(1, len(stat) + 1))
kw = pd.read_csv(C.HANIYA_RESULTS / "differential_expression" / "all_gene_results.csv").set_index("gene")
stat["kruskal_statistic"] = kw["kruskal_statistic"]
stat["kruskal_rank"] = stat["kruskal_statistic"].rank(ascending=False, method="min")
stat["kw_significant"] = kw["significant"]
hani_top50 = list(stat.head(50).index)
hani_top50_kw = list(stat.sort_values("kruskal_statistic", ascending=False).head(50).index)

lr_top50 = list(lr_imp.head(50).index)
rf_top50 = list(rf_imp.head(50).index)
n_tested = len(stat)

bm_rows = []
lr_sel, rf_sel = set(lr_genes), set(rf_genes)
for g in C.HANIYA_BIOMARKERS:
    bm_rows.append({
        "gene": g,
        "in_train_top5000_variance": g in var_genes_train,
        "selected_by_LR_SelectKBest_k1000": g in lr_sel,
        "LR_shap_rank_of_1000": int(lr_imp.loc[g, "rank"]) if g in lr_sel else None,
        "LR_subtype_most_important": lr_imp.loc[g, "subtype_most_important"] if g in lr_sel else None,
        "LR_direction_in_that_subtype": lr_imp.loc[g, "direction_in_that_subtype"] if g in lr_sel else None,
        "selected_by_RF_SelectKBest_k500": g in rf_sel,
        "RF_shap_rank_of_500": int(rf_imp.loc[g, "shap_rank"]) if g in rf_sel else None,
        "RF_mdi_rank_of_500": int(rf_imp.loc[g, "mdi_rank"]) if g in rf_sel else None,
        "Haniya_effect_rank": int(stat.loc[g, "stat_effect_rank"]),
        "Haniya_max_effect_comparison": stat.loc[g, "max_effect_comparison"],
        "in_LR_top50": g in lr_top50, "in_RF_top50": g in rf_top50,
    })
bm = pd.DataFrame(bm_rows)
bm.to_csv(OUT / "haniya_biomarkers_vs_ml.csv", index=False)

ov = {
    "LR top50 ∩ Haniya top50 (max |effect|)": set(lr_top50) & set(hani_top50),
    "RF top50 ∩ Haniya top50 (max |effect|)": set(rf_top50) & set(hani_top50),
    "LR top50 ∩ Haniya top50 (Kruskal H)": set(lr_top50) & set(hani_top50_kw),
    "LR top50 ∩ RF top50": set(lr_top50) & set(rf_top50),
}
ov_df = pd.DataFrame([{"comparison": k, "n_overlap": len(v), "genes": ", ".join(sorted(v))} for k, v in ov.items()])
ov_df["expected_by_chance"] = round(50 * 50 / n_tested, 2)
ov_df.to_csv(OUT / "overlap_summary.csv", index=False)

new = [g for g in lr_top50 if g not in hani_top50]
new_df = pd.DataFrame({
    "gene": new,
    "LR_rank": [int(lr_imp.loc[g, "rank"]) for g in new],
    "LR_subtype_most_important": [lr_imp.loc[g, "subtype_most_important"] for g in new],
    "LR_direction": [lr_imp.loc[g, "direction_in_that_subtype"] for g in new],
    "Haniya_effect_rank": [int(stat.loc[g, "stat_effect_rank"]) if g in stat.index else None for g in new],
    "Haniya_max_abs_effect": [stat.loc[g, "max_abs_effect"] if g in stat.index else None for g in new],
    "Haniya_kw_significant": [bool(stat.loc[g, "kw_significant"]) if g in stat.index else None for g in new],
    "in_PAM50": [g in set() for g in new],  # filled below
})
hani_top50_df = stat.head(50).reset_index()[["gene", "stat_effect_rank", "max_abs_effect", "max_effect_comparison"]]
hani_top50_df.to_csv(OUT / "haniya_top50_statistical_genes.csv", index=False)

# =================================================================== 5. PAM50
PAM50 = ["ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3",
         "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160",
         "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11",
         "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1",
         "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T"]
ALIASES = {"ORC6": ["ORC6L"], "NDC80": ["KNTC2"], "NUF2": ["CDCA1"]}
fs = set(feats)
pam_map = {}
for g in PAM50:
    pam_map[g] = g if g in fs else next((a for a in ALIASES.get(g, []) if a in fs), None)
pam_rows = []
for g, col in pam_map.items():
    pam_rows.append({
        "pam50_gene": g, "dataset_column": col, "in_dataset": col is not None,
        "in_train_top5000_variance": col in var_genes_train,
        "selected_LR_k1000": col in lr_sel, "LR_rank": int(lr_imp.loc[col, "rank"]) if col in lr_sel else None,
        "in_LR_top50": col in lr_top50,
        "selected_RF_k500": col in rf_sel, "RF_shap_rank": int(rf_imp.loc[col, "shap_rank"]) if col in rf_sel else None,
        "in_RF_top50": col in rf_top50,
        "in_Haniya_top50": col in hani_top50,
    })
pam = pd.DataFrame(pam_rows)
pam.to_csv(OUT / "pam50_check.csv", index=False)
pam_cols = set(pam["dataset_column"].dropna())
new_df["in_PAM50"] = new_df["gene"].isin(pam_cols)
new_df.to_csv(OUT / "new_ml_genes_not_in_haniya_top50.csv", index=False)
pam_counts = {
    "PAM50 genes in dataset": int(pam["in_dataset"].sum()),
    "aliases used": {g: c for g, c in pam_map.items() if c and c != g} or "none",
    "in training top-5000 variance": int(pam["in_train_top5000_variance"].sum()),
    "selected by LR (k=1000)": int(pam["selected_LR_k1000"].sum()),
    "in LR top 50": int(pam["in_LR_top50"].sum()),
    "selected by RF (k=500)": int(pam["selected_RF_k500"].sum()),
    "in RF top 50": int(pam["in_RF_top50"].sum()),
    "in Haniya top 50": int(pam["in_Haniya_top50"].sum()),
}
pd.Series(pam_counts).to_csv(OUT / "pam50_counts.csv", header=["value"])

# =================================================================== print key results
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print("\nLR TOP 20 (global mean |SHAP|)\n" + lr_imp.head(20)[["rank", "mean_abs_shap", "subtype_most_important",
                                                              "direction_in_that_subtype"]].round(4).to_string())
print("\nLR TOP 10 PER SUBTYPE (mean |SHAP|, direction)")
for c in CLS:
    s = pcs[pcs.subtype == c].head(10)
    print(f"  {c:<7}" + ", ".join(f"{g}{'↑' if v > 0 else '↓'}" for g, v in zip(s.gene, s.coefficient)))
print("\nLR TOP 8 COEFFICIENTS PER SUBTYPE (+ / -)")
cdf = pd.DataFrame(rows)
for c in CLS:
    p = cdf[(cdf.subtype == c) & (cdf.sign == "positive")].head(8).gene.tolist()
    n = cdf[(cdf.subtype == c) & (cdf.sign == "negative")].head(8).gene.tolist()
    print(f"  {c:<7} + {', '.join(p)}\n  {'':<7} - {', '.join(n)}")
print("\nRF TOP 20 (SHAP | MDI)\n" + rf_imp.head(20)[["shap_rank", "mdi_rank", "mean_abs_shap", "mdi_importance",
                                                     "subtype_most_important"]].round(4).to_string())
if svc_imp is not None:
    print("\nSVC PERMUTATION TOP 15\n" + svc_imp.head(15).round(4).to_string())
print("\nHANIYA 20 BIOMARKERS\n" + bm.to_string(index=False))
print("\nOVERLAPS\n" + ov_df.to_string(index=False))
print("\nNEW (LR top50 not in Haniya top50)\n" + new_df.round(3).to_string(index=False))
print("\nPAM50 COUNTS\n" + "\n".join(f"  {k}: {v}" for k, v in pam_counts.items()))
print("  LR top50 PAM50 genes:", sorted(pam.loc[pam.in_LR_top50, "dataset_column"]))
print("  RF top50 PAM50 genes:", sorted(pam.loc[pam.in_RF_top50, "dataset_column"]))
print("  PAM50 not in top-5000 variance:", sorted(pam.loc[~pam.in_train_top5000_variance, "pam50_gene"]))
print(f"\nSVC: {svc_status}")
print(f"examples: {ex_rows}")
log("PHASE 5 DONE")
