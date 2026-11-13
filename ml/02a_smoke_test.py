"""Phase 2 smoke test (training data only):
  1. unit checks that TopVarianceSelector learns variance from fit-data only
  2. 3-fold CV LogisticRegression, k=1000 -> time per fit, peak RAM, macro F1
  3. pipeline-cache reuse check (second C value should skip preprocessing)
  4. n_jobs=2 memory test
  5. single-fold timings for SVC / RF candidates (for the Phase 3 estimate)
Run from repo root:  python ml/02a_smoke_test.py
"""
import ctypes
import shutil
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC

from ml import config as C
from ml.data import load_split
from ml.pipeline import PIPELINE_CACHE, build_pipeline
from ml.transformers import TopVarianceSelector


# ---------- Windows memory helpers (no extra dependency) ----------
class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
_k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL


def proc_mem_gb():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ok = _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    assert ok, ctypes.get_last_error()
    return c.WorkingSetSize / 1e9, c.PeakWorkingSetSize / 1e9


def sys_avail_gb():
    m = MEMSTAT(); m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9


class SysMemMonitor:
    """Tracks minimum system-available RAM (covers child worker processes too)."""
    def __enter__(self):
        self.start, self.min, self._stop = sys_avail_gb(), sys_avail_gb(), False
        def run():
            while not self._stop:
                self.min = min(self.min, sys_avail_gb()); time.sleep(0.2)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._stop = True; self.t.join()
    @property
    def extra_used(self):
        return self.start - self.min


def section(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}", flush=True)


# ---------- 1. Unit checks ----------
section("1. TopVarianceSelector unit checks")
rng = np.random.default_rng(0)
A = rng.normal(size=(100, 4)).astype(np.float32) * np.array([5, 1, 1, 0.1], dtype=np.float32)
B = rng.normal(size=(50, 4)).astype(np.float32) * np.array([0.1, 1, 1, 5], dtype=np.float32)
sel = TopVarianceSelector(n_top=1).fit(A)
assert sel.get_support(indices=True).tolist() == [0], "should pick col 0 (high var in fit data)"
assert np.allclose(sel.transform(B)[:, 0], B[:, 0]), "transform must reuse fitted indices"
sel_b = TopVarianceSelector(n_top=1).fit(B)
assert sel_b.get_support(indices=True).tolist() == [3]
print("synthetic: fit on A picks col 0; transform(B) returns B[:,0] although B's top-var col is 3  -> OK")

data, feats = load_split(which=("train",))
X, y = data["train"]
print(f"X_train {X.shape}, {X.values.dtype}, {X.values.nbytes / 1e9:.2f} GB in RAM")
base_ws, _ = proc_mem_gb()

cv3 = StratifiedKFold(3, shuffle=True, random_state=C.RANDOM_STATE)
tr_idx, va_idx = next(cv3.split(X, y))
Xn = X.to_numpy()
fold_sel = TopVarianceSelector(5000).fit(X.iloc[tr_idx])
v_tr = Xn[tr_idx].var(axis=0, dtype=np.float64)
v_all = Xn.var(axis=0, dtype=np.float64)
assert np.allclose(fold_sel.variances_, v_tr), "variances must equal fold-train variances"
top_all = set(np.argsort(-v_all, kind="stable")[:5000])
overlap = len(top_all & set(fold_sel.support_idx_))
print(f"real data: selector variances == variance of fold-train rows only -> OK "
      f"(top-5000 differs from all-train top-5000 in {5000 - overlap} genes, as expected)")
assert fold_sel.transform(X.iloc[va_idx]).shape == (len(va_idx), 5000)
print("transform(validation fold) -> (n_val, 5000), uses fitted indices -> OK")
del fold_sel, v_tr, v_all

# ---------- 2. Smoke CV: LR k=1000, 3 folds, n_jobs=1 ----------
section("2. Smoke CV: LogisticRegression, k=1000, 3-fold, n_jobs=1")
shutil.rmtree(PIPELINE_CACHE, ignore_errors=True)
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=C.RANDOM_STATE)
pipe = build_pipeline(lr, k=1000)
gs = GridSearchCV(pipe, {"model__C": [1.0]}, scoring=C.SCORING, cv=cv3, n_jobs=1,
                  refit=False, return_train_score=False, error_score="raise")
with SysMemMonitor() as mon:
    t0 = time.time(); gs.fit(X, y); t_cold = time.time() - t0
r = gs.cv_results_
ws, peak = proc_mem_gb()
print(f"macro F1 (3-fold): {r['mean_test_score'][0]:.4f} ± {r['std_test_score'][0]:.4f}")
print(f"wall {t_cold:.1f}s | per fit: fit {r['mean_fit_time'][0]:.1f}s + score {r['mean_score_time'][0]:.1f}s")
print(f"process RAM: after loading X_train {base_ws:.2f} GB, now {ws:.2f} GB, peak so far {peak:.2f} GB "
      f"| system available RAM: start {mon.start:.2f} GB, min {mon.min:.2f} GB")

# timing breakdown of preprocessing vs model on one fold (no cache)
p = build_pipeline(lr, k=1000, memory=None)
t0 = time.time(); Z = p[:-1].fit_transform(X.iloc[tr_idx], y.iloc[tr_idx]); t_pre = time.time() - t0
t0 = time.time(); LogisticRegression(**lr.get_params()).fit(Z, y.iloc[tr_idx]); t_lr = time.time() - t0
print(f"one-fold breakdown: preprocessing {t_pre:.1f}s, LR fit {t_lr:.1f}s")

# ---------- 3. Cache reuse ----------
section("3. Pipeline cache reuse (new C, same k -> preprocessing should be cached)")
gs2 = GridSearchCV(pipe, {"model__C": [0.1]}, scoring=C.SCORING, cv=cv3, n_jobs=1, refit=False)
t0 = time.time(); gs2.fit(X, y); t_warm = time.time() - t0
print(f"cold run {t_cold:.1f}s vs cached run {t_warm:.1f}s "
      f"(C=0.1 macro F1 {gs2.cv_results_['mean_test_score'][0]:.4f})")
cache_gb = sum(f.stat().st_size for f in PIPELINE_CACHE.rglob("*") if f.is_file()) / 1e9
print(f"cache size on disk: {cache_gb:.2f} GB at {PIPELINE_CACHE}")

# ---------- 4. n_jobs=2 memory test ----------
section("4. n_jobs=2 memory test (uncached, C=0.5)")
gs3 = GridSearchCV(build_pipeline(lr, k=1000, memory=None), {"model__C": [0.5]}, scoring=C.SCORING,
                   cv=cv3, n_jobs=2, pre_dispatch="n_jobs", refit=False)
with SysMemMonitor() as mon2:
    t0 = time.time(); gs3.fit(X, y); t_par = time.time() - t0
print(f"n_jobs=2: wall {t_par:.1f}s, system RAM drawdown {mon2.extra_used:.2f} GB "
      f"(available at start {mon2.start:.2f} GB, min {mon2.min:.2f} GB)")

# ---------- 5. Single-fold timings for other candidates ----------
section("5. Single-fold model timings on preprocessed fold (k=1000 and k=2000)")
yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
pre = {k: build_pipeline(lr, k=k, memory=None)[:-1].fit(X.iloc[tr_idx], yt) for k in (1000, 2000)}
cands = {
    "SVC rbf C=1 (no prob)": lambda: SVC(kernel="rbf", C=1, class_weight="balanced"),
    "SVC linear C=0.01 (no prob)": lambda: SVC(kernel="linear", C=0.01, class_weight="balanced"),
    "SVC rbf C=1 probability=True": lambda: SVC(kernel="rbf", C=1, class_weight="balanced",
                                                probability=True, random_state=C.RANDOM_STATE),
    "LinearSVC C=0.01": lambda: LinearSVC(C=0.01, class_weight="balanced", max_iter=5000),
    "RF 300 trees sqrt n_jobs=1": lambda: RandomForestClassifier(
        n_estimators=300, max_features="sqrt", class_weight="balanced", n_jobs=1,
        random_state=C.RANDOM_STATE),
    "RF 300 trees sqrt n_jobs=2": lambda: RandomForestClassifier(
        n_estimators=300, max_features="sqrt", class_weight="balanced", n_jobs=2,
        random_state=C.RANDOM_STATE),
}
for k, pp in pre.items():
    Zt, Zv = pp.transform(X.iloc[tr_idx]), pp.transform(X.iloc[va_idx])
    for name, make in cands.items():
        if k == 2000 and "n_jobs=2" in name:
            continue
        m = make(); t0 = time.time(); m.fit(Zt, yt); tf = time.time() - t0
        f1 = f1_score(yv, m.predict(Zv), average="macro")
        print(f"k={k:<5} {name:<32} fit {tf:6.1f}s  fold macro F1 {f1:.4f}", flush=True)

_, peak = proc_mem_gb()
print(f"\nOverall process peak RAM: {peak:.2f} GB")
shutil.rmtree(PIPELINE_CACHE, ignore_errors=True)
print("Smoke-test pipeline cache cleared.")
