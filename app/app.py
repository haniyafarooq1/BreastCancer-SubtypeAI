"""Home page. Launch with:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
import streamlit as st  # noqa: E402

cm.setup_page("Home")

st.title("🧬 BreastCancer-SubtypeAI")
st.markdown("#### Predicting PAM50 molecular subtypes of breast cancer from RNA-seq gene expression")
st.error(f"⚠️ **{cm.DISCLAIMER}** This app demonstrates a machine-learning workflow on public research "
         "data. Its predictions must not be used to make decisions about any patient.")

comp = cm.read_csv(str(cm.EVAL / "model_comparison.csv"))
if comp is not None:
    lr = comp.set_index("model").loc["logreg"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary model", "Logistic Regression")
    c2.metric("Test macro F1", f"{lr.test_f1_macro:.3f}")
    c3.metric("Test accuracy", f"{lr.test_accuracy:.3f}")
    c4.metric("Test balanced accuracy", f"{lr.test_balanced_accuracy:.3f}")
    st.caption("Held-out test set of 655 samples, evaluated once. The model was chosen by 5-fold "
               "cross-validation on the training set before testing.")

st.subheader("Objective")
st.markdown("""
- Classify breast tumours into the five **PAM50 intrinsic subtypes** (Basal, Her2, LumA, LumB, Normal)
  from whole-transcriptome RNA-seq expression.
- Compare three models (Logistic Regression, SVM, Random Forest) under a leakage-free pipeline.
- Explain which genes drive the predictions, and compare them with the statistical biomarker
  analysis and with the PAM50 gene list.
- Provide an interactive tool to explore the data, the models and predictions for new samples.
""")

left, right = st.columns([1, 1])
with left:
    st.subheader("Dataset")
    counts = cm.subtype_counts()
    st.markdown(f"""
**GSE96058 (SCAN-B)**: {int(counts.sum()):,} primary breast tumours (technical replicates removed),
30,865 genes, expression as log2(FPKM + 0.1).
""")
    st.dataframe(counts.rename("samples").rename_axis("PAM50 subtype").to_frame(), width="stretch")
    st.caption("Stratified 80/20 split (random_state = 42): 2,618 training / 655 test samples.")
with right:
    st.subheader("Pipeline")
    st.graphviz_chart("""
digraph {
  rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#f0efec", color="#bbbbb5",
  fontname="Helvetica", fontsize=11]; edge [color="#888883"];
  a [label="Expression matrix\\n3,273 samples × 30,865 genes"];
  b [label="Train / test split (80/20, stratified)"];
  c [label="Top-5,000 variance genes (fit on training folds)"];
  d [label="StandardScaler"];
  e [label="SelectKBest (ANOVA F), k tuned"];
  f [label="LR / SVM / RF, class_weight = balanced\\n5-fold CV, macro F1"];
  g [label="One-time test evaluation + SHAP"];
  a -> b -> c -> d -> e -> f -> g;
}
""")
    st.caption("All preprocessing steps sit inside one scikit-learn Pipeline and are re-fit within every "
               "CV fold, so the test set never influences feature selection, scaling or tuning.")

st.info("**Basal-like ≠ triple-negative (TNBC).** *Basal* is a gene-expression (PAM50) subtype. "
        "TNBC is defined clinically by the absence of ER, PR and HER2 protein (IHC/ISH). The two "
        "overlap substantially but are not the same group of patients, so this app never labels a "
        "prediction as TNBC.")

st.subheader("Pages")
st.markdown("""
1. **Dataset Analysis**: subtype distribution, expression distribution, PCA, biomarker heatmap, DE tables
2. **Model Comparison**: test metrics, CV vs test, per-class F1, confusion matrices, selection rule
3. **Subtype Prediction**: upload expression data and get a predicted subtype with probabilities
4. **Explainability**: top genes (SHAP / coefficients), biomarker and PAM50 comparison
""")
cm.footer()
