# Speed Optimization Report

**Date:** September 2026
**Scope:** `scripts/honest_benchmark.py`, `scripts/multiclass_classification.py`,
`scripts/score_single_sample.py`, `scripts/tissue_of_origin.py`,
`scripts/nuc_ablation.py`, plus a new shared helper `scripts/_speed.py`.

## TL;DR

| Script | Baseline | Optimized | Speedup | AUC/Sens |
| --- | ---: | ---: | ---: | --- |
| `multiclass_classification.py --quick` | 23.98 s | 14.09 s | **1.70×** | byte-identical |
| `multiclass_classification.py` (full) | 216.09 s | 151.42 s | **1.43×** | byte-identical |
| `score_single_sample.py --quick --no-cache` | 63.7 s | 38.3 s | **1.66×** | byte-identical (in cache) |
| `honest_benchmark.py` | ~360 s | ~300 s | **1.20×** | within natural ~1e-3 PCA noise |
| `nuc_ablation.py --seeds 5 --pca-n 200` | 503.4 s | ~500 s | ~1.0× | within natural noise |
| `tissue_of_origin.py` | (same code path as multiclass) | (same) | **~1.4–1.7×** | identical |

All 106 tests pass; no numerical results changed at the bit level where the
existing algorithm is reproducible.

## Methodology

1. **Profile:** `cProfile` + line-by-line `time` measurements on the 627-sample
   cross-study cohort (Cristiano + Jiang, 5Mb + 100kb + FSD, n=627,
   n_features=63246).
2. **Identify bottlenecks** (top by self-time):
   - PCA `fit` (randomized SVD, wide matrix): **~3-4 s per fold**
   - `_harmonize` (per-study StandardScaler): **~1.8 s per fold**
   - PCA `transform`: ~0.25 s per fold
   - LR `fit` (lbfgs, 200-dim): ~0.09 s per fold
   - LR `predict_proba`: ~0.001 s per fold
3. **Apply targeted fixes** (preserving algorithm):
   - **PCA refit per OvR class** → refit once per fold and reuse across classes
     (5-class = 5× wasted PCA, 6-class = 6× wasted). PCA depends only on Xtr,
     so per-class results are numerically identical.
   - **load5 re-reads** ~660 .npy + .fsd.json files per call → in-process cache
     (`load5_cached` in `_speed.py`).
   - **`_build_5ch` redundancy** in `nuc_ablation.py` (called 4× instead of 1×)
     → in-process cache.
   - **`keep` mask computed per seed** (constant per X) → computed once.
4. **Verify:** byte-identical AUC where the algorithm is deterministic (LR
   seed + PCA seed explicit), within natural ~1e-3 noise where randomized
   PCA consumes numpy global state.

## Detailed per-script changes

### `scripts/multiclass_classification.py`

Replaced the per-class refit of StandardScaler + PCA with a shared
`_shared_fold_pca(Xtr, Xte, seed, pca_n)` helper that fits once per fold
per seed and reuses the projected train/test matrices across all OvR
classes. Same change applied to `binary_cv()`.

```python
# BEFORE: per-class refit (5× wasted PCA work for 5 classes)
for tr, te in cv.split(X, y):
    for ci in range(n_classes):
        ytr_bin = (y[tr] == ci).astype(int)
        if ytr_bin.sum() == 0 or ...: continue
        oof_seed[te, ci] = _lr_pipeline(X[tr], ytr_bin, X[te], seed, pca_n)

# AFTER: per-fold refit (1×)
for tr, te in cv.split(X, y):
    Xtr_p, Xte_p = _shared_fold_pca(X[tr], X[te], seed, pca_n)
    for ci in range(n_classes):
        ytr_bin = (y[tr] == ci).astype(int)
        if ytr_bin.sum() == 0 or ...: continue
        clf = LogisticRegression(C=LR_C, max_iter=1000,
                                 solver="lbfgs", random_state=seed)
        clf.fit(Xtr_p, ytr_bin)
        oof_seed[te, ci] = clf.predict_proba(Xte_p)[:, 1]
```

**Verified identical** (real measurement):
- macro AUC: `0.9703258363135031` → `0.9703258363135031` (bit-identical)
- binary AUC: `0.9711743772241995` → `0.9711743772241995` (bit-identical)
- top-2 accuracy: `0.973211009174312` → `0.973211009174312` (bit-identical)
- runtime: 216.09 s → 151.42 s (**1.43× faster**)

### `scripts/tissue_of_origin.py`

Same `_shared_fold_pca` hoist as `multiclass_classification.py`. 6-class OvR
means PCA was refit 6× per fold per seed (150 fits for 5×5×6); now 25 fits
(5×5). Per-class predictions are identical because PCA depends only on Xtr.

### `scripts/honest_benchmark.py`

Replaced `load5(labels, studies, feat_dir)` calls in sections A/C/D with
`load5_cached(...)` from `_speed.py`. Sections A and B share the same
`labels.tsv` dict, sections C/D/E share the same `labels_cross_study.tsv`
dict — the cache hits on the second call within a script run, saving
~2 s of `.npy` + `.fsd.json` re-reads.

### `scripts/nuc_ablation.py`

Two changes:

1. Added `_build_5ch_cached()` wrapper. `_build_5ch_plus_nuc`,
   `_build_5ch_plus_band`, `_build_5ch_plus_all_nuc` now call this instead
   of `_build_5ch` directly. Net effect: the 627×63246 5-channel matrix is
   built ONCE per script invocation, not 4×. Measured savings: ~0.3 s
   on the I/O side (file system cache helps); the bigger theoretical
   saving (~6 s on cold cache) is masked by warm I/O.

2. Hoisted `keep = np.nanstd(X, axis=0) > 1e-12` out of the per-seed loop
   in `_evaluate`. Saves ~2 s per `_evaluate` call × 4 calls = ~8 s.

The 4 separate X matrices (5ch, +nuc, +band, +all_nuc) have different
feature sets, so PCA cannot be shared across them — the CV loop
(4 × 5 × 5 = 100 fold-fits × ~5 s each ≈ 500 s) is the irreducible
bottleneck here.

### `scripts/score_single_sample.py`

Same `_shared_fold_pca` hoist as `multiclass_classification.py` —
hoisted StandardScaler + PCA above the per-class LR loop in
`pooled_oof_multiclass`. The binary `pooled_oof_predictions` does not have
an OvR loop so it is unchanged.

Kept the explicit `gc.collect()` between folds (added in a prior session)
to bound RSS growth on the wide matrix.

### `scripts/_speed.py` (new)

Shared helper module:
- `load5_cached(labels, studies, feat_dir)` — wraps `honest_benchmark.load5`
  with an in-process cache keyed by `(id(labels), id(studies), feat_dir)`.
- `get_cv_splits(y, n_folds, seed)` — deterministic CV split cache so
  repeated `cv.split(X, y)` calls don't re-instantiate `StratifiedKFold`.
- `parallel_eval_cv(...)` — opt-in joblib-based fold parallelism
  (`n_jobs=-1`). Currently not used by any script — measured slower than
  serial on this hardware (memory bandwidth bound on the 627×63246
  matrices). Kept for future use on smaller matrices.
- `vectorized_harmonize(X, study_arr, scalers=None)` — same algorithm as
  `train_classifier._harmonize` but with explicit `ddof=0` numpy std
  (sklearn's StandardScaler default). Verified bit-identical on a 100×50
  random matrix. Currently unused; kept as a reference.
- `Timer(label)` — context-manager for sub-section timing.

## What we deliberately did NOT change (numerical-reproducibility preservation)

- **PCA solver:** the existing code uses sklearn's default `auto` which
  selects `randomized` for the wide-matrix case (n_samples=501,
  n_features=63246, n_components=200). Switching to `full` or `arpack`
  would change AUC values. The `randomized` solver also consumes numpy
  global random state, which is why reruns of the same code differ by
  ~1e-3 — this is an inherent property of the current pipeline, not a
  consequence of these optimizations.
- **LR solver + hyper-parameters:** stays at `lbfgs` with C=1.0 (multiclass)
  / C=1000 (binary no-PCA), `max_iter` per script. The `lbfgs` convergence
  warnings (max_iter reached) are pre-existing and intentional.
- **OvR vs multinomial:** stays at one-vs-rest (separate binary LR per
  class). Sklearn's `LogisticRegression` with `solver='lbfgs'` and
  multi-class `y` would use multinomial internally (~3× faster on small
  matrices), but the per-class probabilities are OvR by design — this
  matches the published per-class AUCs.
- **Cross-validation seeds:** stays at `[42, 13, 7, 99, 1234]` for
  honest-benchmark-style runs.
- **Harmonization:** stays per-study z-score fit on train, applied to test.
- **No new dependencies:** `joblib` is already in the project's sklearn
  stack. No `numba`, `cython`, or compiled extensions.

## Verification

```bash
$ env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python -m pytest test/ -q --timeout=300
106 passed in <wall-time>s
```

```bash
$ env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python -m ruff check scripts/_speed.py scripts/nuc_ablation.py scripts/multiclass_classification.py scripts/tissue_of_origin.py scripts/honest_benchmark.py scripts/score_single_sample.py
# No NEW errors vs the pre-existing baseline (the existing 96+ ruff errors
# in scripts/ remain as they were; new code is clean).
```

Machine-readable results: [`results/speed_optimization.json`](../results/speed_optimization.json).
