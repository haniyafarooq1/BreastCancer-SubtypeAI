"""Data loading: primary dataset (float32) + parquet cache + existing train/test split."""
import json
import time

import numpy as np
import pandas as pd

from ml import config as C


def _source_signature():
    st = C.PRIMARY_CSV.stat()
    return {"source": str(C.PRIMARY_CSV), "size": st.st_size, "mtime": st.st_mtime}


def _read_csv():
    header = pd.read_csv(C.PRIMARY_CSV, nrows=0, index_col=0).columns
    # Forcing float32 on every non-target column also acts as the "all numeric" check:
    # a non-numeric value would raise here instead of silently becoming an object column.
    dtypes = {c: np.float32 for c in header if c != C.TARGET}
    dtypes[C.TARGET] = "string"
    dtypes[C.ID_COL] = "string"
    # C engine in row chunks: keeps peak RAM low (laptop has ~8 GB). The pyarrow CSV
    # engine crashed natively (0xC0000409) on this 30k-column file, so it is not used.
    chunks = pd.read_csv(C.PRIMARY_CSV, dtype=dtypes, chunksize=250, low_memory=False)
    df = pd.concat(chunks)
    return df.set_index(C.ID_COL)


def load_primary(use_cache=True, verbose=True):
    """Return the primary dataset: index=sample_id, gene columns float32, plus `subtype`."""
    t0 = time.time()
    meta_path = C.CACHE_PATH.with_suffix(".meta.json")
    sig = _source_signature()
    if use_cache and C.CACHE_PATH.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == sig:
            df = pd.read_parquet(C.CACHE_PATH)
            if verbose:
                print(f"Loaded parquet cache {C.CACHE_PATH} in {time.time() - t0:.1f}s")
            return df
        if verbose:
            print("Cache is stale (source CSV changed); rebuilding")

    df = _read_csv()
    if verbose:
        print(f"Read CSV {C.PRIMARY_CSV.name} in {time.time() - t0:.1f}s", flush=True)
    if use_cache:
        C.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.CACHE_PATH)
        meta_path.write_text(json.dumps(sig))
        if verbose:
            print(f"Wrote parquet cache -> {C.CACHE_PATH}")
    return df


def feature_columns(df):
    """All columns except the target, after checking they are numeric and not metadata."""
    feats = [c for c in df.columns if c != C.TARGET]
    leaked = C.NON_FEATURE_COLS.intersection(feats)
    assert not leaked, f"Metadata columns among features: {leaked}"
    non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    assert not non_numeric, f"Non-numeric feature columns: {non_numeric[:10]}"
    return feats


def _manifest():
    m = pd.read_csv(C.SPLIT_MANIFEST, dtype={C.ID_COL: "string"})
    assert {C.ID_COL, C.TARGET, "split"} <= set(m.columns), m.columns.tolist()
    assert not m[C.ID_COL].duplicated().any(), "Duplicate sample IDs in manifest"
    return m


def load_split(which=("train", "test"), use_cache=True, verbose=True):
    """Return {split: (X, y)} using the existing manifest. Pass which=("train",) during
    training so the test set is never even materialised."""
    df = load_primary(use_cache=use_cache, verbose=verbose)
    feats = feature_columns(df)
    m = _manifest()

    ids = {s: m.loc[m["split"] == s, C.ID_COL].tolist() for s in ("train", "test")}
    assert set(ids["train"]).isdisjoint(ids["test"]), "Train/test overlap!"
    all_ids = ids["train"] + ids["test"]
    assert not any("repl" in i for i in all_ids), "Replicate IDs in manifest"
    assert not df.index.str.contains("repl").any(), "Replicate IDs in dataset"
    missing = set(all_ids) - set(df.index)
    assert not missing, f"{len(missing)} manifest IDs missing from dataset"
    assert C.TARGET not in feats

    # Labels in manifest must agree with dataset labels
    lab = m.set_index(C.ID_COL)[C.TARGET]
    assert (df.loc[lab.index, C.TARGET].astype(str) == lab.astype(str)).all(), "Label mismatch"

    out = {}
    for s in which:
        X = df.loc[ids[s], feats]
        y = df.loc[ids[s], C.TARGET].astype(str)
        out[s] = (X, y)
    return out, feats
