import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
import streamlit as st  # noqa: E402

cm.setup_page("Dataset Analysis", "📊")
st.title("📊 Dataset Analysis")
st.caption("Exploratory analysis by Haniya (scripts `notebooks/09`–`13`). Plots and tables are loaded from "
           "`results/`, not recomputed. Note: these analyses used the full dataset (train + test); they are "
           "descriptive and were not used for model training or feature selection.")

R = cm.RESULTS
t1, t2, t3, t4, t5 = st.tabs(["Subtype distribution", "Expression distribution", "PCA",
                              "Biomarker heatmap", "Differential expression"])
with t1:
    cm.show_image(R / "eda" / "subtype_distribution.png")
    st.markdown("The classes are **imbalanced**: LumA is about half of all samples and Normal is the smallest "
                "class (221). The models therefore use `class_weight='balanced'` and are scored by **macro F1**, "
                "which weights every subtype equally.")
    st.dataframe(cm.subtype_counts().rename("samples").to_frame().T, width="stretch")
with t2:
    cm.show_image(R / "eda" / "expression_distribution.png")
    st.markdown("Expression values are **log2(FPKM + 0.1)**. The spike at about −3.32 = log2(0.1) is "
                "genes with zero expression. Uploaded data for prediction must use the same scale.")
with t3:
    cm.show_image(R / "eda" / "pca_subtypes.png")
    st.markdown("PCA on the 2,000 most variable genes (all samples). PC1 and PC2 explain only about 8.7% "
                "of the variance, so this is a rough view. Along PC1 the subtypes form a gradient from "
                "LumA/LumB (left) through Her2 to Basal (right). Basal sits at one end but overlaps Her2. "
                "LumA, LumB and Normal overlap strongly, which matches the confusions the models make "
                "(Normal → LumA, LumA ↔ LumB).")
with t4:
    cm.show_image(R / "biomarker_candidates" / "biomarker_heatmap.png")
    st.markdown("Median expression per subtype of the 20 exploratory biomarker candidates, standardised per "
                "gene. Luminal genes (ESR1, FOXA1, GATA3, XBP1, MLPH, SPDEF) are high in LumA/LumB and lowest "
                "in Basal. FOXC1 and SOX10 are highest in Basal. The basal keratins KRT5/KRT14 are highest in "
                "Normal (KRT5 is also high in Basal). Proliferation genes (FOXM1, UBE2C, BIRC5, AURKA, PTTG1) "
                "are highest in Basal, intermediate in Her2/LumB, and low in LumA/Normal.")
with t5:
    st.markdown("**Kruskal–Wallis test across the 5 subtypes** (Benjamini–Hochberg FDR). 24,365 of 28,604 "
                "variable genes are significant (adj. p < 0.05). Because many adjusted p-values round to "
                "0.0, genes are ranked here by the **H statistic**.")
    de = cm.table_or_info(R / "differential_expression" / "all_gene_results.csv")
    if de is not None:
        n = st.slider("Number of genes", 10, 100, 25, step=5, key="n_de")
        st.dataframe(de.sort_values("kruskal_statistic", ascending=False).head(n).reset_index(drop=True),
                     width="stretch", hide_index=True)
    st.markdown("**Pairwise comparisons** (Mann–Whitney U, rank-biserial effect size; positive = higher in the "
                "first subtype).")
    files = sorted((R / "biomarker_candidates").glob("*_vs_*.csv"))
    if files:
        choice = st.selectbox("Comparison", [f.stem for f in files], key="pair",
                              format_func=lambda s: s.replace("_vs_", " vs "))
        pw = cm.read_csv(str(R / "biomarker_candidates" / f"{choice}.csv"))
        st.dataframe(pw.head(20)[["gene", "effect_size", "median_difference", "adjusted_p_value"]],
                     width="stretch", hide_index=True)
    else:
        st.info("Pairwise result files not found in `results/biomarker_candidates/`.")
    summ = cm.read_text(str(R / "analysis_summary.md"))
    if summ:
        with st.expander("Haniya's full analysis summary (results/analysis_summary.md)"):
            st.markdown(summ)
cm.footer()
