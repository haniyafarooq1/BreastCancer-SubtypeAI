import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

cm.setup_page("Model Comparison", "⚖️")
st.title("⚖️ Model Comparison")

comp = cm.table_or_info(cm.EVAL / "model_comparison.csv")
if comp is None:
    st.stop()
comp = comp.set_index("model").loc[list(cm.MODEL_NAMES)]
names = {k: v.replace(" (primary)", "") for k, v in cm.MODEL_NAMES.items()}

st.success("**Primary model: Logistic Regression.** It was chosen by a rule written *before* the test set "
           "was used: highest 5-fold CV macro F1, and if the top two are within one CV standard deviation, "
           "choose LR. SVC (0.8844 ± 0.0049) and LR (0.8819 ± 0.0100) were within one SD, so LR was chosen. "
           "SVC is slightly higher on the test set (0.897 vs 0.891, about 1–2 samples per class), but test "
           "results were not used to pick the model.")

st.subheader("Test-set metrics (655 samples, evaluated once)")
cols = {"test_accuracy": "Accuracy", "test_balanced_accuracy": "Balanced acc.",
        "test_precision_macro": "Macro precision", "test_recall_macro": "Macro recall",
        "test_f1_macro": "Macro F1", "test_f1_weighted": "Weighted F1", "test_log_loss": "Log loss"}
t = comp[list(cols)].rename(columns=cols).rename(index=names)
st.dataframe(t.style.format("{:.3f}"), width="stretch")

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("CV vs test macro F1")
    fig = go.Figure()
    fig.add_bar(name="5-fold CV (train)", x=[names[m] for m in comp.index], y=comp["cv_f1_macro_mean"],
                error_y=dict(type="data", array=comp["cv_f1_macro_std"], color="#6b6b66"),
                marker_color="#86b6ef", hovertemplate="%{x}<br>CV macro F1 %{y:.3f}<extra></extra>")
    fig.add_bar(name="Test", x=[names[m] for m in comp.index], y=comp["test_f1_macro"],
                marker_color="#184f95", hovertemplate="%{x}<br>Test macro F1 %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", yaxis=dict(range=[0.75, 0.95], title="Macro F1", gridcolor="#e6e5e1"),
                      height=360, margin=dict(t=10, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.1), bargap=0.3, bargroupgap=0.08)
    st.plotly_chart(fig, width="stretch")
    st.caption("Error bars = CV standard deviation. The y-axis starts at 0.75 to show the small differences.")
with c2:
    st.subheader("Overfitting check")
    o = comp[["train_fit_f1_macro", "cv_f1_macro_mean", "cv_f1_macro_std", "test_f1_macro",
              "test_minus_cv_f1_macro"]].rename(index=names)
    o.columns = ["Train (fit)", "CV mean", "CV SD", "Test", "Test − CV"]
    st.dataframe(o.style.format("{:+.3f}", subset=["Test − CV"]).format(
        "{:.3f}", subset=["Train (fit)", "CV mean", "CV SD", "Test"]), width="stretch")
    st.markdown("For LR and SVC the test score is within about 1 point of the CV estimate, so there is no sign "
                "of overfitting to the training folds. A train-fit F1 near 1.0 is expected with 2.6k samples "
                "and 500–1000 genes. RF is weaker overall.")

st.subheader("Per-class F1: CV vs test")
cvpc = cm.read_csv(str(cm.CV / "cv_per_class_f1.csv"), index_col=0)
tepc = cm.read_csv(str(cm.EVAL / "test_per_class_f1.csv"), index_col=0)
if cvpc is not None and tepc is not None:
    pc = pd.concat({"CV": cvpc.loc[list(cm.MODEL_NAMES)], "Test": tepc.loc[list(cm.MODEL_NAMES)]}, axis=1)
    pc = pc.swaplevel(axis=1)[[(c, s) for c in cm.CLASS_ORDER for s in ("CV", "Test")]].rename(index=names)
    st.dataframe(pc.style.format("{:.3f}"), width="stretch")
    st.caption("Normal is the hardest class (mostly predicted as LumA). Only 44 Normal samples are in the test "
               "set, so one sample moves Normal recall by about 2.3 points.")

st.subheader("Confusion matrices (test set)")
m = st.selectbox("Model", list(cm.MODEL_NAMES), format_func=lambda k: cm.MODEL_NAMES[k], key="cm_model")
cm.show_image(cm.EVAL / f"confusion_matrix_{m}.png")
conf = cm.read_csv(str(cm.EVAL / "main_confusions.csv"))
if conf is not None:
    st.markdown("**Main confusions** (true → predicted, share of the true class):")
    st.dataframe(conf.rename(columns=names), width="stretch", hide_index=True)
st.markdown("""
- **Normal → LumA** is the dominant error. Normal-like samples have a luminal-like profile, partly reflecting
  normal-tissue content.
- **LumA ↔ LumB**: both are ER+ luminal subtypes, separated mainly by proliferation on a continuum.
- **Her2 → LumB**: small on the test set but clearly visible in CV (luminal–HER2 overlap).
- **Basal** is the most separable subtype.
""")

rule = cm.read_text(str(cm.ML / "model_selection_rule.md"))
if rule:
    with st.expander("Model selection rule (written before testing)"):
        st.markdown(rule)
cm.footer()
