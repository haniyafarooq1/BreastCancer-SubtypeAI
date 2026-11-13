"""Automated smoke test of the Streamlit app (streamlit.testing.v1.AppTest).

Loads every page, runs a demo prediction with each model, and checks that no page raises
an exception. Run from the repo root (venv active):
    python app/tests/test_app_smoke.py
"""
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]
PAGES = ["app.py", "pages/1_Dataset_Analysis.py", "pages/2_Model_Comparison.py",
         "pages/3_Subtype_Prediction.py", "pages/4_Explainability.py"]
DISCLAIMER = "Not for clinical diagnosis"
failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def all_text(at):
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption]
    parts += [e.value for e in at.warning] + [e.value for e in at.error] + [e.value for e in at.info]
    parts += [e.value for e in at.sidebar.warning]
    return "\n".join(str(p) for p in parts)


for page in PAGES:
    t = time.time()
    at = AppTest.from_file(str(APP / page), default_timeout=120).run()
    print(f"{page}  ({time.time() - t:.1f}s)")
    check(not at.exception, f"{page}: no exception {[e.value for e in at.exception] if at.exception else ''}")
    check(DISCLAIMER in all_text(at), f"{page}: disclaimer shown")
    if page.startswith("pages/2"):
        for m in ("svc", "rf"):
            at.selectbox(key="cm_model").set_value(m).run()
            check(not at.exception, f"{page}: confusion matrix for {m}")

# ---- demo prediction with every model
pred_page = str(APP / "pages/3_Subtype_Prediction.py")
for m in ("logreg", "svc", "rf"):
    t = time.time()
    at = AppTest.from_file(pred_page, default_timeout=180).run()
    at.selectbox(key="model").set_value(m).run()
    at.button(key="use_demo").click().run()
    print(f"demo prediction [{m}]  ({time.time() - t:.1f}s)")
    check(not at.exception, f"{m}: no exception {[e.value for e in at.exception] if at.exception else ''}")
    check(len(at.error) == 0, f"{m}: no st.error {[e.value for e in at.error]}")
    dfs = [d.value for d in at.dataframe]
    pred_tables = [d for d in dfs if hasattr(d, "columns") and "predicted_subtype" in getattr(d, "columns", [])]
    if not pred_tables:  # styled frames: AppTest returns the underlying data
        pred_tables = [d for d in dfs if "predicted_subtype" in str(getattr(d, "columns", ""))]
    check(len(pred_tables) == 1 and len(pred_tables[0]) == 10, f"{m}: predictions table with 10 samples")
    if pred_tables:
        print("        predicted:", pred_tables[0]["predicted_subtype"].tolist())
    check(any(s.label == "Sample" for s in at.selectbox), f"{m}: sample detail shown")
    # switch sample to exercise the per-sample charts
    at.selectbox(key="sample").select_index(5).run()
    check(not at.exception, f"{m}: per-sample view for another sample")

# ---- input-handling unit checks (same functions the prediction page uses)
sys.path.insert(0, str(APP.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ml import inference as inf  # noqa: E402

print("input handling")
feats = inf.load_feature_genes(); gs = set(feats); stats = inf.load_gene_stats()
used = inf.load_used_genes("logreg")
demo = pd.read_csv(APP / "sample_input" / "demo_samples.csv")
small = demo[["sample_id"] + used]                       # used genes only
p = inf.parse_expression_table(small, gs)
check(p.data is not None and p.data.shape == (10, len(used)), "used-genes-only file parses (10 x 1000)")
mi = inf.build_model_input(p.data, used, feats, stats["median"])
check(mi.X is not None and mi.X.shape == (10, len(feats)) and not mi.missing_used, "builds full 30,865-col input")
w, _ = inf.sanity_check(mi.X[used], stats)
check(not w, "no scale warning for real log2(FPKM+0.1) data")
mi2 = inf.build_model_input(p.data.drop(columns=used[:3]), used, feats, stats["median"])
check(mi2.X is None and mi2.missing_used == used[:3], "refuses and lists 3 missing required genes")
bad = p.data.copy(); bad.iloc[0, 0] = np.nan
check(inf.build_model_input(bad, used, feats, stats["median"]).X is None, "refuses empty value in required gene")
raw_counts = (2 ** p.data - 0.1) * 40                    # un-logged, scaled -> count-like
w, _ = inf.sanity_check(raw_counts, stats)
check(any("raw counts" in x for x in w), "warns on unlogged / count-like values")
t = small.set_index("sample_id").T.reset_index().rename(columns={"index": "gene"})
pt = inf.parse_expression_table(t, gs)
check(pt.data is not None and pt.data.shape == (10, len(used))
      and any("transposed" in m for _, m in pt.messages), "genes-as-rows file detected and transposed")
extra = small.assign(NOT_A_GENE=1.0)
mi3 = inf.build_model_input(inf.parse_expression_table(extra, gs).data, used, feats, stats["median"])
check(mi3.X is not None and "NOT_A_GENE" in mi3.extra_cols, "extra columns ignored with a note")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURE(S)'}")
sys.exit(1 if failures else 0)
