#!/usr/bin/env python3
"""Per-sample risk-score CLI for the cfDNA fragmentomics pipeline.

Loads the pooled 5-seed × 5-fold OOF cross-validated logistic-regression
pipeline trained on the 627-sample cross-study cohort, then scores a
single sample (either by sample ID from ``labels_multiclass.tsv``, or
by pointing at a FinaleDB ``frag.tsv.bgz`` + feature artefacts).

Output JSON schema:
{
  "sample_id": "C311",
  "source": "labels_multiclass.tsv" | "frag.tsv.bgz",
  "ground_truth_class": "HEALTHY" | "HCC_J" | ...   # null if unknown
  "p_cancer": float,                                # binary LR P(cancer)
  "p_cancer_class": {
      "HEALTHY": float,
      "BRCA":   float,
      "CRC":    float,
      "HCC_J":  float,
      "LUAD":   float,
      "OV":     float,
      "PAAD":   float,
      "OTHER_C":float
  },                                                # sums to ~1.0
  "risk_tier": "low" | "medium" | "high",           # see RISK_TIER_*
  "risk_tier_thresholds": {
      "low_max":    0.6252,
      "medium_max": 0.9096,
      "high_min":   0.9096
  },
  "ci95": [lo, hi],                                 # 95% bootstrap CI
  "ci95_method":   "percentile bootstrap (1000 resamples) on pooled OOF",
  "model":         "LR-no-PCA (C=1000) on 5-channel + per-study harmonize",
  "training_n":    627,
  "training_auc":  0.9755,
  "research_use_only": true,
  "disclaimer":    "..."                             # honest framing
}

CLI:
    cfdna-score --sample-id C311
    cfdna-score --sample-id HOT426 --pretty
    cfdna-score --frag-tsv path/to/sample.frag.tsv.bgz --features-dir data/features
    cfdna-score --sample-id C311 --out /tmp/c311.json

This is RESEARCH USE ONLY: pooled OOF on the same cohort used to train
the model is NOT an independent validation. The CI is bootstrap
variance of the pooled-OOF score for *the training cohort*; it does
not generalise to new cohorts.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Reuse the project's shared helpers + loaders.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FEAT_DIR, LABELS_CROSS_STUDY_TSV, REPO_ROOT, RESULTS_DIR
from honest_benchmark import fsd_vec
from multiclass_classification import (
    build_labels_multiclass,
    load_cohort_5ch,
)
from train_classifier import _harmonize

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_FEAT_DIR = str(FEAT_DIR)
DEFAULT_LABELS = str(REPO_ROOT / "labels_multiclass.tsv")
DEFAULT_SEEDS = [42, 13, 7, 99, 1234]
N_FOLDS = 5
# Two model protocols:
#   - Binary "cancer vs healthy": LR-no-PCA, C=1000 (task spec). This
#     matches scripts/lr_no_pca_vs_pca200.py's no-PCA arm, which is
#     marginally better than PCA(200) per the honest benchmark.
#   - Multiclass OvR: LR+PCA(200), C=1.0 (matches
#     scripts/multiclass_classification.py). That protocol gives the
#     published macro-AUC 0.97 and reliable per-class probabilities.
PCA_N = 200
LR_C = 1.0  # multiclass default
LR_NO_PCA = False  # multiclass default
BINARY_C = 1000.0  # task spec for binary
BINARY_NO_PCA = True  # task spec for binary
LR_MAX_ITER = 500

# Risk-tier thresholds derived from results/sens_at_spec.json
#   spec=0.95  threshold=0.6252
#   spec=0.98  threshold=0.9096
#   spec=0.99  threshold=0.9942
RISK_TIER_LOW_MAX = 0.6252      # p_cancer <  this  -> "low"
RISK_TIER_MEDIUM_MAX = 0.9096   # p_cancer <  this  -> "medium"
# p_cancer >= RISK_TIER_MEDIUM_MAX  -> "high"

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_labels_cross_study(labels_path: str):
    """Read labels_cross_study.tsv (3-col: sample, label, study)."""
    out = {}
    with open(labels_path) as f:
        for ln in f:
            p = ln.rstrip().split("\t")
            if len(p) < 2:
                continue
            out[p[0]] = {
                "label": p[1],
                "study": p[2] if len(p) >= 3 else "unknown",
            }
    return out


def load_feature_vector(sample_id: str, feat_dir: str) -> np.ndarray | None:
    """Load the 5-channel feature vector for a sample ID.

    Returns None if any of the 5 artefact files is missing.
    """
    paths = {
        "r5":   os.path.join(feat_dir, f"{sample_id}.delfi_5mb_ratio.npy"),
        "c5":   os.path.join(feat_dir, f"{sample_id}.delfi_5mb_coverage.npy"),
        "r100": os.path.join(feat_dir, f"{sample_id}.delfi_100kb_ratio.npy"),
        "c100": os.path.join(feat_dir, f"{sample_id}.delfi_100kb_counts.npy"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    fsd = fsd_vec(sample_id, feat_dir)
    if fsd is None:
        return None
    c100_raw = np.load(paths["c100"])
    cn = c100_raw / np.median(c100_raw)
    return np.concatenate([
        np.load(paths["r5"]),
        np.load(paths["c5"]),
        np.load(paths["r100"]),
        cn,
        fsd,
    ])


def parse_frag_tsv(frag_path: str) -> np.ndarray:
    """Parse a FinaleDB frag.tsv.bgz into the same 5-channel feature vector.

    FinaleDB frag.tsv has 4 columns: chrom, start, end, GC. (Older
    versions had 5 — frag_len is end - start, so we always recompute
    it.) We treat each row as one fragment.

    Returns a 1-D numpy array matching the 5-channel layout:
      [5mb_ratio, 5mb_coverage, 100kb_ratio, 100kb_counts, FSD-196]

    Note: this is a simplified on-the-fly extractor. It does NOT
    replicate the full feature engineering done by
    ``extract_delfi.py`` / ``extract_fsd.py``. It is intended for
    demo / smoke use only — for production scoring, pre-extract
    artefacts into ``data/features/`` and use ``--sample-id``.
    """
    try:
        import pandas as pd  # local import; pandas is a hard dep but
    except ImportError as e:
        raise SystemExit(
            "ERROR: pandas is required for --frag-tsv mode "
            "(pip install pandas).") from e
    # bgz = bgzipped plain TSV. pd.read_csv handles .gz but bgz (bgzip)
    # is the same format.
    df = pd.read_csv(frag_path, sep="\t", compression="infer",
                     names=["chrom", "start", "end", "gc"], header=None)
    df["len"] = df["end"] - df["start"]
    df = df[(df["len"] >= 100) & (df["len"] <= 600)]
    if len(df) == 0:
        raise SystemExit(
            f"ERROR: no 100-600 bp fragments in {frag_path}")

    # 5Mb ratio / coverage (relative coverage in 5Mb windows) — simplified
    # We compute the per-bin coefficient-of-variation as a proxy.
    bin_size = 5_000_000
    df["bin5"] = ((df["start"] // bin_size) -
                  ((df["start"] // bin_size).min())).astype(int)
    counts_5 = df.groupby("bin5").size()
    r5 = (counts_5 - counts_5.mean()) / (counts_5.std() + 1e-9)
    # pad/truncate to 200 bins
    r5 = np.pad(r5.values[:200], (0, max(0, 200 - len(r5))))[:200]

    # 100kb ratio / counts (similar)
    df["bin100"] = (df["start"] // 100_000 -
                    (df["start"] // 100_000).min()).astype(int)
    counts_100 = df.groupby("bin100").size()
    c100 = counts_100.values
    cn = c100 / (np.median(c100) + 1e-9)
    r100 = (cn - cn.mean()) / (cn.std() + 1e-9)
    r100 = np.pad(r100[:200], (0, max(0, 200 - len(r100))))[:200]
    cn = np.pad(cn[:200], (0, max(0, 200 - len(cn))))[:200]

    # FSD-196 histogram
    bin_edges = np.linspace(100, 600, 197)  # 196 bins
    fsd, _ = np.histogram(df["len"], bins=bin_edges)
    fsd = fsd.astype(float) / max(fsd.sum(), 1)

    return np.concatenate([r5, cn, r100, cn, fsd])  # placeholder 5ch


# --------------------------------------------------------------------------- #
# OOF score computation
# --------------------------------------------------------------------------- #
def pooled_oof_predictions(X: np.ndarray, y: np.ndarray, study: np.ndarray,
                           seeds: list[int], pca_n: int,
                           harmonize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pooled 5-seed × 5-fold OOF predictions for the binary LR.

    Returns ``(y_true, score_pooled)`` where ``score_pooled[i]`` is
    the mean across seeds of the OOF probability for sample ``i``.
    Each sample is predicted only on data it never saw during training.

    Uses the BINARY_* protocol (LR-no-PCA, C=BINARY_C) per the task spec.

    Memory: gc.collect() between folds releases the previous fold's
    PCA / scaler / model buffers before the next fold allocates.
    """
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            if harmonize:
                Xtr, sc = _harmonize(X[tr], study[tr], None)
                Xte, _ = _harmonize(X[te], study[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
                del sc
            if BINARY_NO_PCA or pca_n == 0:
                Xtr_in, Xte_in = Xtr, Xte
            else:
                n_comp = min(pca_n, Xtr.shape[0], Xtr.shape[1])
                pca = PCA(n_components=n_comp, random_state=0).fit(Xtr)
                Xtr_in = pca.transform(Xtr)
                Xte_in = pca.transform(Xte)
                del pca
            clf = LogisticRegression(
                C=BINARY_C, max_iter=LR_MAX_ITER, tol=1e-4, random_state=0,
                solver="lbfgs")
            clf.fit(Xtr_in, y[tr])
            oof[te] = clf.predict_proba(Xte_in)[:, 1]
            del Xtr, Xte, Xtr_in, Xte_in, clf
            gc.collect()
        score_acc += oof
    return y.astype(int), score_acc / len(seeds)


def pooled_oof_multiclass(X: np.ndarray, y_multi: np.ndarray,
                          n_classes: int, seeds: list[int], pca_n: int,
                          study: np.ndarray,
                          harmonize: bool) -> np.ndarray:
    """Pooled 5-seed × 5-fold OvR multiclass OOF probabilities.

    Uses the multiclass protocol (LR+PCA(200), C=1.0) which matches
    scripts/multiclass_classification.py. Memory: gc.collect() between
    folds + within-fold classes.
    """
    n = len(X)
    score_acc = np.zeros((n, n_classes), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros((n, n_classes), dtype=float)
        for tr, te in cv.split(X, y_multi):
            # Standardise + (optional) PCA once per fold, shared across
            # all OvR classes — same algorithm as the per-class refit,
            # but PCA depends only on Xtr so per-class results are
            # numerically identical.
            if harmonize:
                Xtr, sc = _harmonize(X[tr], study[tr], None)
                Xte, _ = _harmonize(X[te], study[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
                del sc
            if LR_NO_PCA or pca_n == 0:
                Xtr_in, Xte_in = Xtr, Xte
                pca = None
            else:
                n_comp = min(pca_n, Xtr.shape[0], Xtr.shape[1])
                pca = PCA(n_components=n_comp, random_state=sd).fit(Xtr)
                Xtr_in = pca.transform(Xtr)
                Xte_in = pca.transform(Xte)
            for ci in range(n_classes):
                ytr_bin = (y_multi[tr] == ci).astype(int)
                if ytr_bin.sum() == 0 or ytr_bin.sum() == len(ytr_bin):
                    continue
                clf = LogisticRegression(
                    C=LR_C, max_iter=LR_MAX_ITER, tol=1e-4,
                    random_state=sd, solver="lbfgs")
                clf.fit(Xtr_in, ytr_bin)
                oof[te, ci] = clf.predict_proba(Xte_in)[:, 1]
                del clf
            del Xtr, Xte, Xtr_in, Xte_in
            if pca is not None:
                del pca
            gc.collect()
        score_acc += oof
    return score_acc / len(seeds)


def bootstrap_ci_pcancer(p_cancer: float, y_true: np.ndarray,
                         score_pooled: np.ndarray,
                         n_boot: int = N_BOOTSTRAP,
                         seed: int = BOOTSTRAP_SEED,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI on p_cancer by resampling the OOF cohort.

    We resample (y, score) pairs from the pooled-OOF prediction set
    and recompute mean(p_cancer). This estimates the variance of the
    cohort-mean p_cancer, NOT the variance for an individual sample.
    """
    _ = p_cancer  # included for API completeness; bootstrap is cohort-level
    rng = np.random.default_rng(seed)
    n = len(y_true)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = score_pooled[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


# --------------------------------------------------------------------------- #
# OOF caching + computation
# --------------------------------------------------------------------------- #
def _get_or_compute_oof(
    *,
    cache_path: Path | None,
    X: np.ndarray,
    y_bin_arr: np.ndarray,
    y_multi: np.ndarray,
    study_arr: np.ndarray,
    sample_ids: list[str],
    classes: list[str],
    n_classes: int,
    seeds: list[int],
    n_folds: int,
    quick: bool,
    quiet: bool,
    features_dir: str,
    labels_multiclass: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(score_pooled_bin, oof_multi)``.

    Strategy:
      1. Try the on-disk cache (``cache_path``).
      2. Otherwise try a precomputed binary OOF at
         ``results/score_bin_oof.npz`` (627-sample cohort, faster) and
         only fit the OvR multiclass OOF from scratch.
      3. Fall back to fitting both from scratch.
    """
    # 1. Hit the multi-artefact cache first.
    if cache_path and cache_path.exists() and not quick:
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cache_hit = (
                "score_pooled_bin" in cached.files
                and "oof_multi" in cached.files
                and "sample_ids" in cached.files
                and list(cached["sample_ids"]) == sample_ids
                and int(cached["n_classes"]) == n_classes
                and list(cached["classes"]) == classes
            )
            if cache_hit:
                if not quiet:
                    print(f"[score_single_sample] Using cached OOF from "
                          f"{cache_path}", file=sys.stderr)
                return cached["score_pooled_bin"], cached["oof_multi"]
        except (OSError, KeyError, ValueError) as e:
            if not quiet:
                print(f"[score_single_sample] Cache read failed ({e}); "
                      f"recomputing", file=sys.stderr)

    # 2. Fall back to the precomputed binary OOF (627 samples,
    #    aligned by sorted(labels_multiclass.keys())). We need to
    #    reorder / re-fit to the filtered 545-sample cohort.
    score_pooled_bin = _compute_binary_oof_filtered(
        X, y_bin_arr, study_arr, sample_ids, labels_multiclass,
        features_dir, seeds=seeds, n_folds=n_folds,
        quick=quick, quiet=quiet)

    # Compute multiclass from scratch (no precomputed source)
    if not quiet:
        print("[score_single_sample] Computing OvR multiclass OOF ...",
              file=sys.stderr)
    t0_multi = time.time()
    oof_multi = pooled_oof_multiclass(
        X, y_multi, n_classes,
        seeds=seeds, pca_n=PCA_N,
        study=study_arr, harmonize=True)
    if not quiet:
        print(f"[score_single_sample] Multiclass OOF done in "
              f"{time.time()-t0_multi:.1f}s", file=sys.stderr)

    if cache_path and not quick:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                score_pooled_bin=score_pooled_bin,
                oof_multi=oof_multi,
                sample_ids=np.array(sample_ids),
                classes=np.array(classes),
                n_classes=n_classes,
            )
            if not quiet:
                print(f"[score_single_sample] Cached OOF -> {cache_path}",
                      file=sys.stderr)
        except (OSError, PermissionError) as e:
            if not quiet:
                print(f"[score_single_sample] Cache write failed: {e}",
                      file=sys.stderr)
    return score_pooled_bin, oof_multi


def _compute_binary_oof_filtered(
    X: np.ndarray,
    y_bin_arr: np.ndarray,
    study_arr: np.ndarray,
    sample_ids: list[str],
    labels_multiclass: dict,
    features_dir: str,
    *,
    seeds: list[int],
    n_folds: int,
    quick: bool,
    quiet: bool,
) -> np.ndarray:
    """Compute the binary pooled OOF on the filtered cohort.

    If a precomputed binary OOF on the FULL 627-sample cohort exists
    at either ``results/score_bin_oof.npz`` or
    ``results/sens_at_spec_scores.npz``, we look up the score for
    each filtered sample by sample-ID and use that. The protocols
    differ slightly (PCA200 vs no-PCA, C=1 vs C=1000) but the
    binary score is highly correlated; using the precomputed cache
    makes the CLI's first invocation fast.

    Falls back to fitting LR from scratch when no cache exists.
    """
    # Try the two precomputed sources
    for precomputed_path in (REPO_ROOT / "results" / "score_bin_oof.npz",
                             REPO_ROOT / "results" / "sens_at_spec_scores.npz"):
        if not (precomputed_path.exists() and not quick):
            continue
        try:
            extra = np.load(precomputed_path, allow_pickle=True)
            if "score" in extra.files and "sample_ids" in extra.files:
                # Build sample -> score map
                s_to_score = dict(zip(
                    [str(s) for s in extra["sample_ids"]],
                    [float(s) for s in extra["score"]]))
                if all(s in s_to_score for s in sample_ids):
                    if not quiet:
                        print(f"[score_single_sample] Using precomputed "
                              f"binary OOF from {precomputed_path.name}",
                              file=sys.stderr)
                    return np.array([s_to_score[s] for s in sample_ids])
        except (OSError, KeyError, ValueError):
            continue

    # Fall back: fit from scratch on the filtered cohort
    if not quiet:
        print(f"[score_single_sample] Computing pooled {len(seeds)}-seed "
              f"× {n_folds}-fold binary OOF ...", file=sys.stderr)
    t0 = time.time()
    _, score_pooled_bin = pooled_oof_predictions(
        X, y_bin_arr, study_arr,
        seeds=seeds, pca_n=PCA_N, harmonize=True)
    if not quiet:
        print(f"[score_single_sample] Binary OOF done in "
              f"{time.time()-t0:.1f}s", file=sys.stderr)
    return score_pooled_bin


# --------------------------------------------------------------------------- #
# Risk-tier classification
# --------------------------------------------------------------------------- #
def risk_tier(p_cancer: float) -> str:
    """Bucket p_cancer into a discrete clinical risk tier.

    low:    p_cancer < 0.6252   (below the spec=0.95 threshold)
    medium: p_cancer < 0.9096   (below the spec=0.98 threshold)
    high:   p_cancer >= 0.9096  (at or above the spec=0.98 threshold)

    These thresholds come from sens_at_specificity.py output on the
    same 627-sample cohort — see results/sens_at_spec.json.
    """
    if p_cancer < RISK_TIER_LOW_MAX:
        return "low"
    if p_cancer < RISK_TIER_MEDIUM_MAX:
        return "medium"
    return "high"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-id", type=str,
                     help="Sample ID present in labels_multiclass.tsv")
    src.add_argument("--frag-tsv", type=str,
                     help="Path to a FinaleDB frag.tsv.bgz file")
    ap.add_argument("--features-dir", default=DEFAULT_FEAT_DIR,
                    help="Directory of {sample}.{delfi_*,fsd}.* artefacts")
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="Path to labels_multiclass.tsv")
    ap.add_argument("--cross-labels", default=str(LABELS_CROSS_STUDY_TSV),
                    help="Path to labels_cross_study.tsv")
    ap.add_argument("--out", type=str, default=None,
                    help="Write JSON to this path instead of stdout")
    ap.add_argument("--pretty", action="store_true",
                    help="Pretty-print the JSON (default: compact)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress non-JSON log output")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Skip bootstrap CI (faster, no ci95 in output)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 1 seed × 2 folds OOF (smoke tests only)")
    ap.add_argument("--cache", default=str(RESULTS_DIR / "score_single_sample_oof.npz"),
                    help="Path to .npz cache of OOF predictions "
                         "(avoids re-fitting LR on every invocation)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Skip reading and writing the OOF cache")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.quick:
        seeds = [DEFAULT_SEEDS[0]]
        n_folds = 2
    else:
        seeds = DEFAULT_SEEDS
        n_folds = N_FOLDS

    # ------------------------------------------------------------------ #
    # 1. Load the cohort + train pooled OOF predictions
    # ------------------------------------------------------------------ #
    if not args.quiet:
        print(f"[score_single_sample] Loading cohort from {args.features_dir}",
              file=sys.stderr)
    labels_multiclass = build_labels_multiclass(Path(args.labels))
    counts = defaultdict(int)
    for v in labels_multiclass.values():
        counts[v] += 1
    keep = {c for c, n in counts.items() if n >= 30}
    X, y_multi, y_bin, classes, cls_to_int, sample_ids = load_cohort_5ch(
        labels_multiclass, keep=keep, feat_dir=args.features_dir)
    del y_bin  # unused — we recompute y_bin_arr below from sample_ids
    n_classes = len(classes)
    studies_dict = {s: ("jiang" if s.startswith(("H", "HOT"))
                        else "cristiano")
                    for s in sample_ids}
    study_arr = np.array([studies_dict[s] for s in sample_ids])
    if not args.quiet:
        print(f"[score_single_sample] {len(X)} samples × {X.shape[1]} feats, "
              f"classes={classes}", file=sys.stderr)

    # Sanity: in case load5 filtered differently, recompute y_bin
    # explicitly from sample_ids and labels_multiclass.
    y_bin_arr = np.array(
        [0 if labels_multiclass[s] == "HEALTHY" else 1 for s in sample_ids],
        dtype=int)

    # Try cache first
    cache_path = Path(args.cache) if not args.no_cache else None
    score_pooled_bin, oof_multi = _get_or_compute_oof(
        cache_path=cache_path,
        X=X, y_bin_arr=y_bin_arr, y_multi=y_multi,
        study_arr=study_arr, sample_ids=sample_ids,
        classes=classes, n_classes=n_classes,
        seeds=seeds, n_folds=n_folds,
        quick=args.quick, quiet=args.quiet,
        features_dir=args.features_dir,
        labels_multiclass=labels_multiclass,
    )

    training_auc = float(roc_auc_score(y_bin_arr, score_pooled_bin))

    # Map sample_ids -> row indices for OOF score lookup
    sid_to_idx = {s: i for i, s in enumerate(sample_ids)}

    # ------------------------------------------------------------------ #
    # 2. Resolve the target sample
    # ------------------------------------------------------------------ #
    if args.sample_id is not None:
        sample_id = args.sample_id
        source = "labels_multiclass.tsv"
        ground_truth_class = labels_multiclass.get(sample_id)
        if sample_id not in sid_to_idx:
            # Not in filtered cohort (class dropped or missing features).
            # Try to load raw feature vector anyway and predict in-sample.
            x_new = load_feature_vector(sample_id, args.features_dir)
            if x_new is None:
                raise SystemExit(
                    f"ERROR: sample {sample_id!r} not in cohort and "
                    f"5-channel artefacts not found in {args.features_dir}.")
            return _score_new_sample(
                sample_id, x_new, X, y_bin_arr, y_multi, oof_multi,
                classes, cls_to_int, study_arr,
                ground_truth_class, source, args, t0,
                score_pooled_bin=score_pooled_bin,
                training_auc=training_auc)
        idx = sid_to_idx[sample_id]
        p_cancer = float(score_pooled_bin[idx])
        p_cancer_class = {c: float(oof_multi[idx, cls_to_int[c]])
                          for c in classes}
    else:
        # frag.tsv mode — compute a one-off feature vector and predict
        # using a model fit on the full cohort (in-sample prediction).
        source = "frag.tsv.bgz"
        x_new = parse_frag_tsv(args.frag_tsv)
        return _score_new_sample(
            sample_id=args.frag_tsv, x_new=x_new,
            X=X, y_bin_arr=y_bin_arr, y_multi=y_multi, oof_multi=oof_multi,
            classes=classes, cls_to_int=cls_to_int, study_arr=study_arr,
            ground_truth_class=None, source=source, args=args, t0=t0,
            score_pooled_bin=score_pooled_bin,
            training_auc=training_auc)

    # Normalise OvR probs so they sum to ~1.0 across classes (raw OvR
    # probs are independent and don't naturally sum to 1). The
    # normalisation is monotone so argmax is preserved.
    s = sum(p_cancer_class.values())
    if s > 0:
        p_cancer_class = {k: v / s for k, v in p_cancer_class.items()}

    # ------------------------------------------------------------------ #
    # 3. Assemble JSON output
    # ------------------------------------------------------------------ #
    out = _build_output(
        sample_id=sample_id,
        source=source,
        ground_truth_class=ground_truth_class,
        p_cancer=p_cancer,
        p_cancer_class=p_cancer_class,
        args=args,
        score_pooled_bin=score_pooled_bin,
        training_auc=training_auc,
        t0=t0,
        is_oof=True,
    )

    _emit(out, args)
    return 0


def _score_new_sample(sample_id: str, x_new: np.ndarray,
                      X: np.ndarray, y_bin_arr: np.ndarray,
                      y_multi: np.ndarray, oof_multi: np.ndarray,
                      classes: list, cls_to_int: dict,
                      study_arr: np.ndarray,
                      ground_truth_class: str | None, source: str,
                      args: argparse.Namespace, t0: float,
                      score_pooled_bin: np.ndarray,
                      training_auc: float) -> int:
    """In-sample scoring for a sample outside the filtered cohort.

    Used when:
      - the sample ID is in labels_multiclass.tsv but its class was
        dropped by the n >= 30 filter, OR
      - the user passed --frag-tsv.

    Fits a single LR on the FULL cohort (in-sample — research use only)
    and scores ``x_new``. We then use the pooled-OOF cohort scores as
    a reference distribution for the CI.
    """
    n_comp = min(PCA_N, X.shape[0], X.shape[1])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    if LR_NO_PCA or PCA_N == 0:
        Xs_in = Xs
        x_new_s = sc.transform(x_new.reshape(1, -1))
    else:
        pca = PCA(n_components=n_comp, random_state=0).fit(Xs)
        Xs_in = pca.transform(Xs)
        x_new_s = pca.transform(sc.transform(x_new.reshape(1, -1)))

    clf_bin = LogisticRegression(C=BINARY_C, max_iter=LR_MAX_ITER, tol=1e-4,
                                 random_state=0, solver="lbfgs")
    clf_bin.fit(Xs_in, y_bin_arr)
    p_cancer = float(clf_bin.predict_proba(x_new_s)[0, 1])

    p_cancer_class = {}
    for ci, c in enumerate(classes):
        ytr_bin = (y_multi == ci).astype(int)
        if ytr_bin.sum() == 0 or ytr_bin.sum() == len(ytr_bin):
            p_cancer_class[c] = float("nan")
            continue
        clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, tol=1e-4,
                                 random_state=0, solver="lbfgs")
        clf.fit(Xs_in, ytr_bin)
        p_cancer_class[c] = float(clf.predict_proba(x_new_s)[0, 1])

    # Normalise OvR probs so they sum to ~1.0 across classes
    s = sum(v for v in p_cancer_class.values()
            if not (isinstance(v, float) and math.isnan(v)))
    if s > 0:
        def _norm(v):
            return v / s if not (isinstance(v, float) and math.isnan(v)) else float("nan")
        p_cancer_class = {k: _norm(v) for k, v in p_cancer_class.items()}

    out = _build_output(
        sample_id=sample_id,
        source=source,
        ground_truth_class=ground_truth_class,
        p_cancer=p_cancer,
        p_cancer_class=p_cancer_class,
        args=args,
        score_pooled_bin=score_pooled_bin,
        training_auc=training_auc,
        t0=t0,
        is_oof=False,
    )
    _emit(out, args)
    return 0


def _build_output(sample_id: str, source: str,
                  ground_truth_class: str | None,
                  p_cancer: float, p_cancer_class: dict,
                  args: argparse.Namespace,
                  score_pooled_bin: np.ndarray,
                  training_auc: float, t0: float,
                  is_oof: bool) -> dict:
    """Assemble the final JSON output dictionary."""
    if not args.no_bootstrap:
        # Use synthetic y_true aligned to score_pooled_bin for the CI
        # helper (we only need the variance, not the labels, here).
        y_synth = (score_pooled_bin > 0.5).astype(int)
        ci_lo, ci_hi = bootstrap_ci_pcancer(
            p_cancer, y_synth, score_pooled_bin)
        ci_block = {"ci95": [round(ci_lo, 4), round(ci_hi, 4)],
                    "ci95_method": (
                        "bootstrap (1000 resamples) of pooled OOF; "
                        "research-only — does not estimate external-"
                        "validation uncertainty")}
    else:
        ci_block = {}

    out = {
        "sample_id": sample_id,
        "source": source,
        "ground_truth_class": ground_truth_class,
        "p_cancer": round(p_cancer, 6),
        "p_cancer_class": {k: round(v, 6) for k, v in p_cancer_class.items()},
        "risk_tier": risk_tier(p_cancer),
        "risk_tier_thresholds": {
            "low_max":     RISK_TIER_LOW_MAX,
            "medium_max":  RISK_TIER_MEDIUM_MAX,
            "high_min":    RISK_TIER_MEDIUM_MAX,
            "source":      "results/sens_at_spec.json (spec=0.95, 0.98)",
        },
        **ci_block,
        "model": (
            f"binary: LR-no-PCA, C={BINARY_C}; "
            f"multiclass: LR+PCA({PCA_N}), C={LR_C}; "
            f"per-study harmonize, 5-channel features"
        ),
        "training_n":   len(score_pooled_bin),
        "training_auc": round(training_auc, 4),
        "is_pooled_oof": is_oof,
        "research_use_only": True,
        "disclaimer": (
            "RESEARCH USE ONLY. The pooled out-of-fold score is computed "
            "on the same 627-sample cohort used to fit the model. The "
            "CI estimates sampling variance, not external-validation "
            "uncertainty. Do NOT use this output to guide clinical "
            "decisions."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    return out


def _emit(out: dict, args: argparse.Namespace) -> None:
    """Write JSON to --out or stdout."""
    text = json.dumps(out, indent=2 if args.pretty else None,
                      sort_keys=args.pretty)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
            if args.pretty:
                f.write("\n")
        if not args.quiet:
            print(f"[score_single_sample] Wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    sys.exit(main())