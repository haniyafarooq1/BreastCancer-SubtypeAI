import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

cm.setup_page("Explainability", "🧠")
st.title("🧠 Explainability")
st.info("**How to read this page.** SHAP values and coefficients show *associations the model uses* to "
        "reproduce PAM50 labels, not causal effects. Correlated genes can share or swap importance. "
        "The genes are **exploratory**, from a single cohort, and **not clinically validated biomarkers**.")

E = cm.EXPL
t1, t2, t3, t4 = st.tabs(["Logistic Regression (primary)", "Random Forest markers",
                          "Comparison with statistics", "PAM50 check"])

# ------------------------------------------------------------------ LR
with t1:
    st.warning("**Basal caveat:** the L2-regularised LR spreads its Basal weight across many correlated genes. "
               "In the Basal beeswarm, almost all of the signal sits in the 'sum of other features' term, so "
               "single-gene LR ranks for Basal are flat and not very informative. See the **Random Forest "
               "markers** tab for classic Basal markers (FOXC1, KRT5, KRT14, FOXA1↓, ESR1↓).")
    st.subheader("Global importance")
    cm.show_image(E / "lr_shap_global_top20.png",
                  "Mean |SHAP| on the 655 test samples (log-odds), split by how much each subtype contributes.")

    st.subheader("Top genes per subtype")
    pcs = cm.table_or_info(E / "lr_top_genes_per_subtype.csv")
    if pcs is not None:
        n = st.slider("Genes per subtype", 5, 15, 10, key="n_pcs")
        cols = st.columns(5)
        for col, c in zip(cols, cm.CLASS_ORDER):
            s = pcs[pcs.subtype == c].head(n)
            col.markdown(f"**{c}**")
            col.dataframe(pd.DataFrame({"gene": s.gene, "dir": ["↑" if v > 0 else "↓" for v in s.coefficient]}),
                          width="stretch", hide_index=True, height=35 * (n + 1) + 3)
        st.caption("Ranked by class-specific mean |SHAP|. ↑ = higher expression pushes towards the subtype, "
                   "↓ = away, relative to the other subtypes (multinomial model).")
    coefs = cm.read_csv(str(E / "lr_top_coefficients_per_subtype.csv"))
    if coefs is not None:
        with st.expander("Top 15 positive / negative coefficients per subtype"):
            c = st.selectbox("Subtype", cm.CLASS_ORDER, key="coef_sub")
            a, b = st.columns(2)
            pos = coefs[(coefs.subtype == c) & (coefs.sign == "positive")]
            neg = coefs[(coefs.subtype == c) & (coefs.sign == "negative")]
            a.markdown(f"**Higher expression → towards {c}**")
            a.dataframe(pos[["rank", "gene", "coefficient"]], hide_index=True, width="stretch")
            b.markdown(f"**Higher expression → away from {c}**")
            b.dataframe(neg[["rank", "gene", "coefficient"]], hide_index=True, width="stretch")

    st.subheader("Per-subtype SHAP beeswarm")
    sub = st.selectbox("Subtype", cm.CLASS_ORDER, key="bees_sub")
    cm.show_image(E / f"lr_shap_beeswarm_{sub}.png",
                  "Each dot is one test sample; colour = log2 expression of the gene (blue low, red high).")

    st.subheader("Example explanations (waterfall)")
    w1, w2 = st.columns(2)
    with w1:
        cm.show_image(E / "lr_waterfall_basal_correct_Basal.png", "Correctly classified Basal sample (F2815).")
    with w2:
        which = st.radio("Misclassified Normal sample (F2606, predicted LumA): explain",
                         ["LumA", "Normal"], horizontal=True, key="wf_class")
        cm.show_image(E / f"lr_waterfall_normal_misclassified_as_luma_{which}.png",
                      f"Contributions to the log-odds of {which}.")

# ------------------------------------------------------------------ RF
with t2:
    st.markdown("Random Forest concentrates its importance on classic luminal / basal marker genes, which "
                "complements the more diffuse LR view.")
    cm.show_image(E / "rf_top20_genes.png", "Left: SHAP (TreeExplainer, 200 stratified test samples, "
                  "probability scale). Right: built-in impurity importance (MDI), which is biased towards "
                  "high-variance features.")
    rf = cm.read_csv(str(E / "rf_importance.csv"))
    if rf is not None:
        st.dataframe(rf.head(30)[["gene", "shap_rank", "mdi_rank", "mean_abs_shap", "mdi_importance",
                                  "subtype_most_important"]], hide_index=True, width="stretch")
    st.caption("SVC: KernelExplainer was skipped as too slow, and test-set permutation importance was "
               "skipped because it was estimated at about 29 minutes.")

# ------------------------------------------------------------------ Haniya comparison
with t3:
    st.markdown("Haniya's statistical analysis used the **full dataset**, so it is used here only for "
                "comparison, never for model training. Her genes are ranked by the maximum |rank-biserial "
                "effect size| over the 10 pairwise subtype comparisons.")
    st.subheader("Haniya's 20 biomarker candidates in the models")
    bm = cm.table_or_info(E / "haniya_biomarkers_vs_ml.csv")
    if bm is not None:
        show = bm.rename(columns={
            "in_train_top5000_variance": "top-5000 var", "selected_by_LR_SelectKBest_k1000": "LR selected",
            "LR_shap_rank_of_1000": "LR rank /1000", "LR_subtype_most_important": "LR subtype",
            "LR_direction_in_that_subtype": "LR dir", "selected_by_RF_SelectKBest_k500": "RF selected",
            "RF_shap_rank_of_500": "RF SHAP rank /500", "RF_mdi_rank_of_500": "RF MDI rank",
            "Haniya_effect_rank": "Haniya rank", "Haniya_max_effect_comparison": "Haniya strongest contrast",
            "in_LR_top50": "LR top 50", "in_RF_top50": "RF top 50"})
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("All 20 pass the variance filter and are selected by LR; 17/20 are selected by RF. "
                   "4 are in the LR top 50 (ESR1, BIRC5, PTTG1, BCL2) and 15 are in the RF top 50.")
    st.subheader("Top-50 overlap")
    ov = cm.read_csv(str(E / "overlap_summary.csv"))
    if ov is not None:
        st.dataframe(ov, hide_index=True, width="stretch")
    st.markdown("""
**Why the LR overlap is low (5/50) while RF's is high (25/50):** Haniya's top 50 is dominated by the most extreme
pairwise contrasts, *Basal vs LumA* (25 genes) and *LumB vs Normal* (22). These are the easy boundaries. The LR
top 50 focuses on the hard ones: 19 genes matter most for LumA, 17 for LumB, 8 for Her2, 5 for Normal and only
1 for Basal.
""")
    new = cm.read_csv(str(E / "new_ml_genes_not_in_haniya_top50.csv"))
    if new is not None:
        with st.expander(f"{len(new)} genes in the LR top 50 but not in Haniya's top 50"):
            st.dataframe(new, hide_index=True, width="stretch")
            st.caption("None are statistically new: all are significant in her Kruskal–Wallis test, just ranked "
                       "lower by single-gene effect size (median rank 961). ERBB2 and GRB7 mark only Her2, so "
                       "their maximum pairwise effect is moderate, yet they are key for a multi-class model.")

# ------------------------------------------------------------------ PAM50
with t4:
    st.markdown("Comparison with the **PAM50** gene list (Parker et al., *J Clin Oncol* 2009). All 50 genes are "
                "present in the dataset under their current names (ORC6, NDC80, NUF2; no aliases needed).")
    counts = cm.read_csv(str(E / "pam50_counts.csv"), index_col=0)
    if counts is not None:
        st.dataframe(counts, width="stretch")
    st.markdown("15 PAM50 genes are in the LR top 50, against about 2 expected by chance from its 1,000 features. "
                "16 are in the RF top 50. 9 PAM50 genes (ACTR3B, BAG1, BLVRA, CCNB1, CXXC5, GPR160, MDM2, MYC, "
                "TYMS) are removed by the variance pre-filter; their information is likely carried by "
                "correlated genes.")
    st.error("**Limitation:** the SCAN-B PAM50 labels were themselves assigned from the expression of these "
             "genes. A high PAM50 overlap and high accuracy are therefore partly built in: the models learn "
             "to reproduce an expression-based classifier, not to predict an independent clinical outcome.")
    pam = cm.read_csv(str(E / "pam50_check.csv"))
    if pam is not None:
        with st.expander("Per-gene PAM50 table"):
            st.dataframe(pam, hide_index=True, width="stretch")

cm.footer()
