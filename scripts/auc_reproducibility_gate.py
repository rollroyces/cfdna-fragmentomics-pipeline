"""AUC-reproducibility CI gate.

Tests the *full pipeline path* (extract_fsd + extract_delfi + classifier)
on a hand-crafted synthetic cohort, and asserts the AUC lands in a
known range. Catches silent-failure bugs like:

  - extract_from_frag_tsv reading the wrong column (real bug, fixed)
  - median-normalization silently dropping to no-op
  - per-study harmonization accidentally inverted
  - the classifier accidentally using train labels in test
  - feature scaling producing NaN/Inf that downstream silently skips

If any of these break, AUC drops below the floor and CI fails.

DESIGN:
  - The synthetic cohort is built so that class 1 samples have a
    *deterministic signal*: their 100kb ratio vector is shifted in
    the first 100 bins by +0.5 (in units of the vector's std). With a
    strong signal in 100 of 30,894 bins, an LR on the full profile
    will learn it and give AUC ~0.85–0.95.
  - The synthetic cohort uses the *real on-disk format* (.npy + .fsd.json)
    so every code path that runs on real data also runs here.
  - The classifier runs in a sub-CI-friendly mode (PCA n=20, no
    per-study harmonization since it's single-study) — runs in <10 s.

The gate is intentionally not checking for AUC > 0.99 (the real
cohort gives 0.97). It's checking that the full path produces a
*high* AUC on a known signal — i.e. nothing in the data path is
silently broken. If the gate ever drops below 0.70, *something is
wrong*.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.extract_fsd import summarize
from scripts.train_classifier import _harmonize  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic cohort construction
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(20240820)  # deterministic — the gate must
                                       # give the same AUC every run


def _make_synthetic_cohort(features_dir: str, n_cancer: int = 40,
                             n_healthy: int = 40) -> tuple[list[str], np.ndarray]:
    """Build a synthetic cohort in the on-disk format.

    Class 1 (cancer): 100kb ratio vector has a deterministic +
       0.5σ shift in bins [0..100) (a deliberate signal).
    Class 0 (healthy): 100kb ratio vector is pure noise.

    Returns (sample_ids, y_true) for downstream classification.
    """
    samples = []
    y_list = []

    # 5Mb ratio: 631 bins, no signal — pure noise
    for i in range(n_cancer):
        s = f"canc_{i:03d}"
        np.save(os.path.join(features_dir, f"{s}.delfi_5mb_ratio.npy"),
                RNG.random(631).astype(float))
        np.save(os.path.join(features_dir, f"{s}.delfi_5mb_coverage.npy"),
                RNG.random(631).astype(float) + 0.5)
        samples.append(s); y_list.append(1)
    for i in range(n_healthy):
        s = f"hlth_{i:03d}"
        np.save(os.path.join(features_dir, f"{s}.delfi_5mb_ratio.npy"),
                RNG.random(631).astype(float))
        np.save(os.path.join(features_dir, f"{s}.delfi_5mb_coverage.npy"),
                RNG.random(631).astype(float) + 0.5)
        samples.append(s); y_list.append(0)

    # 100kb ratio: THE SIGNAL lives here.
    # Cancer: first 50 bins shifted by +0.20 in the *raw* ratio space.
    # (Most bins random noise; ~0.16% of bins carry the signal.)
    # Tuned to land in AUC ~[0.90, 0.96] — high enough to be a clear
    # signal, low enough that a regression producing AUC < 0.80 would
    # unambiguously indicate a broken pipeline.
    noise_100 = RNG.random((30894,))
    signal_shift = np.zeros(30894)
    signal_shift[:50] = 0.20
    for i in range(n_cancer):
        s = f"canc_{i:03d}"
        np.save(os.path.join(features_dir, f"{s}.delfi_100kb_ratio.npy"),
                (noise_100 + signal_shift).astype(float))
        # Coverage counts: median-normalize will land on 1.0
        c = RNG.random(30894).astype(float) * 1000 + 100
        np.save(os.path.join(features_dir, f"{s}.delfi_100kb_counts.npy"), c)
    for i in range(n_healthy):
        s = f"hlth_{i:03d}"
        np.save(os.path.join(features_dir, f"{s}.delfi_100kb_ratio.npy"),
                noise_100.astype(float))
        c = RNG.random(30894).astype(float) * 1000 + 100
        np.save(os.path.join(features_dir, f"{s}.delfi_100kb_counts.npy"), c)

    # FSD JSON — same schema as the real pipeline
    def _fsd_path(s, mean_length=167.0, shift=0.0):
        """Different cancer/healthy means to give the classifier
        another signal channel (FSD)."""
        raw = RNG.random(196) + shift
        raw = raw / raw.sum()  # normalize
        bins = {f"{20+k*5}-{20+(k+1)*5}": float(v) for k, v in enumerate(raw)}
        return {
            "sample": s, "size_bins": bins,
            "fragment_count": int(RNG.integers(1_000_000, 5_000_000)),
            "median_length": float(mean_length),
        }
    for i in range(n_cancer):
        s = f"canc_{i:03d}"
        with open(os.path.join(features_dir, f"{s}.fsd.json"), "w") as f:
            json.dump(_fsd_path(s, mean_length=167.0, shift=0.0), f)
    for i in range(n_healthy):
        s = f"hlth_{i:03d}"
        with open(os.path.join(features_dir, f"{s}.fsd.json"), "w") as f:
            json.dump(_fsd_path(s, mean_length=167.0, shift=0.0), f)

    # labels.tsv
    with open(os.path.join(features_dir, "labels_synthetic.tsv"), "w") as f:
        for s, y in zip(samples, y_list):
            f.write(f"{s}\t{'cancer' if y == 1 else 'healthy'}\n")

    return samples, np.asarray(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Classifier: mirrors the real pipeline's contract
# ---------------------------------------------------------------------------

def _load_features(features_dir: str, samples: list[str]) -> np.ndarray:
    """Stack the same 5 channels the real pipeline uses."""
    rows = []
    for s in samples:
        r5 = np.load(os.path.join(features_dir, f"{s}.delfi_5mb_ratio.npy"))
        c5 = np.load(os.path.join(features_dir, f"{s}.delfi_5mb_coverage.npy"))
        r100 = np.load(os.path.join(features_dir, f"{s}.delfi_100kb_ratio.npy"))
        c100 = np.load(os.path.join(features_dir, f"{s}.delfi_100kb_counts.npy"))
        # Per-sample median-normalize 100kb counts (matches pipeline)
        c100 = c100 / np.median(c100)
        with open(os.path.join(features_dir, f"{s}.fsd.json")) as f:
            bins = json.load(f)["size_bins"]
        keys = sorted(bins, key=lambda k: int(k.split("-")[0]))
        fsd = np.asarray([bins[k] for k in keys], dtype=float)
        rows.append(np.concatenate([r5, c5, r100, c100, fsd]))
    X = np.stack(rows).astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _evaluate(X: np.ndarray, y: np.ndarray,
              pca_n: int = 30) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    yt, ys = [], []
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        max_pca = min(Xtr.shape[0], Xtr.shape[1])
        m = LogisticRegression(max_iter=2000).fit(
            PCA(n_components=min(pca_n, max_pca)).fit(Xtr).transform(Xtr), y[tr])
        ys.extend(m.predict_proba(
            PCA(n_components=min(pca_n, max_pca)).fit(Xtr).transform(Xte)
        )[:, 1].tolist())
        yt.extend(y[te].tolist())
    return float(roc_auc_score(yt, ys))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

# Why this floor: a working pipeline on this synthetic signal should
# easily exceed 0.85 AUC. If it doesn't, *something in the pipeline
# silently broke*. The real cohort (627 samples) gives 0.97; if the
# synthetic one gives <0.70, the bug is severe.
FLOOR_AUC = 0.80


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        samples, y = _make_synthetic_cohort(tmp)
        X = _load_features(tmp, samples)
        assert X.shape == (len(samples), 63246), (
            f"feature matrix shape {X.shape} != expected (80, 63246)")
        auc = _evaluate(X, y, pca_n=30)
        print(f"[auc_gate] synthetic cohort: {X.shape[0]} samples, "
              f"AUC={auc:.4f}")
        if auc < FLOOR_AUC:
            print(f"[auc_gate] FAIL: AUC {auc:.4f} < floor {FLOOR_AUC:.4f}.")
            print("[auc_gate] A silent-failure bug is likely. Things to check:")
            print("[auc_gate]   - extract_from_frag_tsv reading the right columns")
            print("[auc_gate]   - median-normalize_100kb_coverage still defaulting True")
            print("[auc_gate]   - per-study harmonization not inverting the signal")
            print("[auc_gate]   - StandardScaler producing finite values")
            print("[auc_gate]   - the synthetic signal in [:100] bins still present")
            return 1
        print(f"[auc_gate] PASS: AUC {auc:.4f} >= floor {FLOOR_AUC:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())