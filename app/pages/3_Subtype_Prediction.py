import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ml import inference as inf  # noqa: E402

cm.setup_page("Subtype Prediction", "🔬")
st.title("🔬 Subtype Prediction")
st.warning(f"⚠️ **{cm.DISCLAIMER}** Predictions are only meaningful for bulk-tumour RNA-seq processed like "
           "SCAN-B (log2(FPKM + 0.1)). The model has not been validated on other cohorts or platforms.")

# ------------------------------------------------------------------ model + requirements
model = st.selectbox("Model", list(cm.MODEL_NAMES), format_func=lambda k: cm.MODEL_NAMES[k], key="model")
used = cm.used_genes(model)
feats = cm.feature_genes()
gene_set = set(feats)
stats = cm.gene_stats()

st.markdown(f"This model needs **{len(used)} genes** (the genes kept by its feature selection). Other genes "
            f"are not needed: they are filled with the training-set median, which does not change the "
            f"prediction.")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    up = st.file_uploader("Upload a CSV (samples as rows, gene symbols as columns; optional `sample_id` "
                          "column; a genes-as-rows file is detected and transposed automatically)",
                          type=["csv", "tsv", "txt"], key="upload")
with c2:
    st.write("")
    template = pd.DataFrame(columns=["sample_id"] + used)
    st.download_button(f"⬇️ Template CSV ({len(used)} genes)", template.to_csv(index=False),
                       file_name=f"template_{model}_{len(used)}_genes.csv", mime="text/csv", key="template")
    st.caption("Header row only: one row per sample, values in log2(FPKM + 0.1).")
with c3:
    st.write("")
    demo_clicked = st.button("🧪 Use demo samples", key="use_demo",
                             help="10 held-out test samples (2 per subtype), labels removed")

# the most recent action wins: clicking demo, or uploading a new file
if demo_clicked:
    st.session_state["source"] = "demo"
if up is not None:
    up_id = getattr(up, "file_id", up.name)
    if up_id != st.session_state.get("last_upload_id"):
        st.session_state["last_upload_id"] = up_id
        st.session_state["source"] = "upload"
elif st.session_state.get("source") == "upload":
    st.session_state["source"] = None


def read_any(buf, name):
    text = buf.getvalue() if hasattr(buf, "getvalue") else buf.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", errors="replace")
    sep = "\t" if name.lower().endswith((".tsv", ".txt")) or text.split("\n", 1)[0].count("\t") > \
        text.split("\n", 1)[0].count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


@st.cache_data(show_spinner="Reading demo samples ...")
def read_demo():
    return pd.read_csv(cm.DEMO_FILE)


raw, source_name = None, None
if st.session_state.get("source") == "upload" and up is not None:
    try:
        raw, source_name = read_any(up, up.name), up.name
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
elif st.session_state.get("source") == "demo":
    if cm.DEMO_FILE.exists():
        raw, source_name = read_demo(), "demo_samples.csv (10 test samples, labels removed)"
    else:
        st.error(f"Demo file not found: `{cm.rel(cm.DEMO_FILE)}`")

if raw is None:
    st.info("Upload a file or click **Use demo samples** to start.")
    cm.footer()
    st.stop()

# ------------------------------------------------------------------ parse + validate
st.subheader("Input check")
parsed = inf.parse_expression_table(raw, gene_set)
LEVEL = {"info": st.info, "warning": st.warning, "error": st.error}
for lvl, msg in parsed.messages:
    LEVEL[lvl](msg)
if parsed.data is None:
    cm.footer(); st.stop()
df = parsed.data
st.markdown(f"**Source:** {source_name}. **{len(df)} sample(s)**, {df.shape[1]} columns, "
            f"{len(set(df.columns) & set(used))}/{len(used)} required genes present.")

mi = inf.build_model_input(df, used, feats, stats["median"])
for lvl, msg in mi.messages:
    LEVEL[lvl](msg)
if mi.missing_used:
    miss = pd.DataFrame({"missing or invalid required gene": mi.missing_used})
    with st.expander(f"Show the {len(mi.missing_used)} missing/invalid required genes", expanded=True):
        st.dataframe(miss, width="stretch", hide_index=True, height=min(400, 35 * (len(miss) + 1)))
        st.download_button("⬇️ Download list", miss.to_csv(index=False), "missing_genes.csv", key="dl_missing")
if mi.X is None:
    cm.footer(); st.stop()

warns, frac_out = inf.sanity_check(mi.X[used], stats)
if warns:
    for w in warns:
        st.warning("Data-scale check: " + w)
else:
    st.success(f"Data-scale check passed: values look like log2(FPKM + 0.1) "
               f"({frac_out:.1%} of values outside the training range ± 2).")

# ------------------------------------------------------------------ predict
pipe = cm.load_pipeline(model)
pred, proba, classes = inf.predict(pipe, mi.X)
classes = list(classes)
res = pd.DataFrame(proba, index=mi.X.index, columns=[f"P({c})" for c in classes])
res.insert(0, "confidence", proba.max(axis=1))
res.insert(0, "predicted_subtype", pred)
res.insert(2, "model", cm.MODEL_NAMES[model])

st.subheader("Predictions")
st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c.startswith("P(") or c == "confidence"}),
             width="stretch")
st.download_button("⬇️ Download predictions (CSV)", res.to_csv(), f"predictions_{model}.csv",
                   mime="text/csv", key="dl_pred")
st.caption("Probabilities are calibrated within the training cohort. A low maximum probability (e.g. < 0.6) "
           "means the sample lies between subtypes. Basal predictions refer to the PAM50 Basal-like subtype, "
           "not to TNBC.")

# ------------------------------------------------------------------ per-sample view
st.subheader("Sample detail")
sid = st.selectbox("Sample", list(res.index), key="sample")
i = list(res.index).index(sid)
left, right = st.columns([1, 1.25])
with left:
    st.metric("Predicted subtype", pred[i], f"probability {proba[i].max():.2f}", delta_color="off")
    fig = go.Figure(go.Bar(
        x=proba[i], y=classes, orientation="h", marker_color=[cm.SUBTYPE_COLORS[c] for c in classes],
        text=[f"{p:.1%}" for p in proba[i]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 1.12], title="Probability", tickformat=".0%", gridcolor="#e6e5e1"),
                      yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10, b=10, l=10, r=10),
                      plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig, width="stretch", key="proba_chart")
with right:
    if model == "logreg":
        phi, base, genes = inf.lr_contributions(pipe, mi.X.iloc[[i]], cm.lr_background_mean())
        ci = classes.index(pred[i])
        contrib = pd.Series(phi[0, :, ci], index=genes)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)
        rest = contrib.sum() - top.sum()
        expr = mi.X.iloc[i][top.index]
        st.markdown(f"**Top genes for the predicted class ({pred[i]})**: SHAP contributions to the log-odds of "
                    f"{pred[i]} (red = pushes towards {pred[i]}, blue = away)")
        fig2 = go.Figure(go.Bar(
            x=top.values[::-1], y=[f"{g} ({expr[g]:.1f})" for g in top.index[::-1]], orientation="h",
            marker_color=[cm.TOWARDS if v > 0 else cm.AWAY for v in top.values[::-1]],
            hovertemplate="%{y}<br>contribution %{x:+.3f}<extra></extra>"))
        fig2.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="white",
                           xaxis=dict(title="SHAP value (log-odds)", gridcolor="#e6e5e1", zeroline=True,
                                      zerolinecolor="#888883"))
        st.plotly_chart(fig2, width="stretch", key="shap_chart")
        st.caption(f"Gene (log2 expression). Baseline log-odds {base[ci]:+.2f}; these 12 genes sum to "
                   f"{top.sum():+.2f}; the other {len(genes) - 12} genes sum to {rest:+.2f}. LR spreads weight "
                   f"over many correlated genes, so the 'other genes' term is often large, especially for "
                   f"Basal. These are model associations, not causal effects.")
    else:
        st.info("Per-sample gene contributions are shown for the Logistic Regression model (exact, fast SHAP). "
                "For SVC/RF, see the global explanations on the **Explainability** page.")
cm.footer()
