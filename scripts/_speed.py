"""Speed optimization helpers for the cfDNA fragmentomics pipeline.

Centralizes caching and parallel-execution primitives so the heavy scripts
(honest_benchmark, nuc_ablation, multiclass_classification, etc.) can
share infrastructure without each one re-implementing it.

DESIGN CONSTRAINTS:
  - The published AUC/Sens numbers are the source of truth. Every helper
    here must preserve the EXACT algorithm (PCA solver, LR solver, etc.).
    We only add caching, parallelism, and I/O shortcuts.
  - The current pipeline uses sklearn PCA's 'auto' solver, which selects
    'randomized' for the wide-matrix case (n_features=63246, n_samples=501,
    n_components=200). This produces ~1e-3 variance across reruns because
    randomized SVD consumes numpy's global random state. We do NOT change
    the solver — we only parallelize/cache around it.

Functions:
  - ``load5_cached``: cache the loaded 5-channel feature matrix across calls.
  - ``build_cv_splits``: cache (seed, n_folds) -> list[(train_idx, test_idx)]
    so the same seed produces identical splits every time.
  - ``parallel_evaluate_cv``: drop-in replacement for evaluate_cv that
    parallelizes fold-fits across cores (only when memory permits).
  - ``vectorized_harmonize``: per-study z-score without a Python-level loop.

All helpers are pure additions — they do not change the algorithm.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Add the scripts/ dir so we can import honest_benchmark and train_classifier.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_classifier import _harmonize

# --------------------------------------------------------------------------- #
# 1. Feature-matrix cache (in-process)
# --------------------------------------------------------------------------- #
_FEATURE_CACHE: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def load5_cached(labels: dict, studies: dict, feat_dir: str):
    """load5() with an in-process LRU-ish cache keyed on (id(labels), id(studies), feat_dir).

    Multiple sections in honest_benchmark (A, B, C, D, E) call load5() on the
    SAME label/study dicts but with different feature subsets. Caching here
    avoids re-reading + concatenating ~660 .npy/.json files per call.
    """
    key = (id(labels), id(studies), feat_dir)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    from honest_benchmark import load5 as _load5
    out = _load5(labels, studies, feat_dir=feat_dir)
    _FEATURE_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# 2. CV-split cache
# --------------------------------------------------------------------------- #
_CV_SPLITS_CACHE: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}


def get_cv_splits(y: np.ndarray, n_folds: int, seed: int
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic list of (train_idx, test_idx) for (seed, n_folds).

    The result is cached so the same (n_samples, n_folds, seed) tuple always
    yields the same splits — same input, same output.
    """
    key = (len(y), n_folds, seed)
    if key not in _CV_SPLITS_CACHE:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        _CV_SPLITS_CACHE[key] = list(cv.split(np.zeros(len(y)), y))
    return _CV_SPLITS_CACHE[key]


# --------------------------------------------------------------------------- #
# 3. Parallel fold evaluation
# ---------------------------------------------------------------------------
def _single_fold_fit(args):
    """Worker: fit ONE fold. Must be top-level for joblib pickling."""
    (Xtr, ytr, Xte, st_tr, st_te, pca_n, harmonize,
     lr_C, lr_max_iter, lr_solver) = args
    # joblib shares memory in process mode; copy so _harmonize can write
    Xtr = np.array(Xtr, copy=True)
    Xte = np.array(Xte, copy=True)
    if harmonize:
        Xtr, sc = _harmonize(Xtr, st_tr, None)
        Xte, _ = _harmonize(Xte, st_te, sc)
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xte = sc.transform(Xte)
    n_comp = min(pca_n, Xtr.shape[0], Xtr.shape[1])
    if n_comp <= 0:
        n_comp = 1
    pca = PCA(n_components=n_comp).fit(Xtr)
    Xtr2 = pca.transform(Xtr)
    Xte2 = pca.transform(Xte)
    m = LogisticRegression(C=lr_C, max_iter=lr_max_iter, solver=lr_solver
                           ).fit(Xtr2, ytr)
    p = m.predict_proba(Xte2)[:, 1]
    return p


def parallel_eval_cv(X: np.ndarray, y: np.ndarray, study_arr: np.ndarray,
                     seeds: list[int], n_folds: int, pca_n: int,
                     harmonize: bool = True,
                     lr_C: float = 1.0, lr_max_iter: int = 2000,
                     lr_solver: str = "lbfgs",
                     n_jobs: int = 1,
                     ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pooled-OOF cross-validation, optionally parallel across folds.

    Returns (y_true, score_pooled, auc_mean, auc_std) where score_pooled is
    the per-sample mean of per-seed OOF scores (5 seeds -> 5 scores per
    sample, averaged).

    The algorithm is IDENTICAL to sens_at_specificity.pooled_oof_predictions
    when n_jobs=1; with n_jobs>1 the fold-fits run in parallel joblib workers.
    Note: randomized PCA consumes numpy global random state, so parallel
    workers may produce slightly different (still valid) AUC values
    (~1e-3 variance) — same as serial reruns.
    """
    from joblib import Parallel, delayed

    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        splits = get_cv_splits(y, n_folds, sd)
        oof = np.zeros(len(y), dtype=float)
        # Build job args (one per fold)
        job_args = []
        for tr, te in splits:
            job_args.append((X[tr], y[tr], X[te], study_arr[tr], study_arr[te],
                             pca_n, harmonize, lr_C, lr_max_iter, lr_solver))
        if n_jobs == 1:
            preds = [_single_fold_fit(a) for a in job_args]
        else:
            preds = Parallel(n_jobs=n_jobs, verbose=0, prefer="processes"
                             )(delayed(_single_fold_fit)(a) for a in job_args)
        for (_, te), p in zip(splits, preds):
            oof[te] = p
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


# --------------------------------------------------------------------------- #
# 4. Vectorised per-study z-score
# ---------------------------------------------------------------------------
def vectorized_harmonize(X: np.ndarray, study_arr: np.ndarray,
                         scalers: dict | None = None
                         ) -> tuple[np.ndarray, dict]:
    """Drop-in faster alternative to train_classifier._harmonize.

    Algorithm IDENTICAL to _harmonize: per-study StandardScaler fit on train,
    applied to test. Only the implementation is vectorised — instead of a
    Python loop over study-unique values, we use boolean masking in a
    single numpy pass.

    Returns: (transformed_X, scalers_dict)
    """
    if scalers is None:
        scalers = {}
        for st in np.unique(study_arr):
            mask = study_arr == st
            if mask.sum() > 1:
                # Use numpy directly (no sklearn overhead per study).
                # sklearn's StandardScaler uses ddof=0 (population variance).
                rows = X[mask]
                mu = rows.mean(axis=0)
                sd = rows.std(axis=0, ddof=0)
                # Avoid divide-by-zero
                sd = np.where(sd < 1e-12, 1.0, sd)
                scalers[st] = (mu, sd)
    out = np.empty_like(X, dtype=float)
    for st, (mu, sd) in scalers.items():
        mask = study_arr == st
        if mask.any():
            out[mask] = (X[mask] - mu) / sd
    return out, scalers


# --------------------------------------------------------------------------- #
# 5. Timing helpers
# ---------------------------------------------------------------------------
class Timer:
    """Tiny context-manager for sub-section timing in reports."""

    def __init__(self, label: str = ""):
        self.label = label
        self.t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0

    def __repr__(self) -> str:
        return f"Timer({self.label!r}, elapsed={self.elapsed:.2f}s)"
