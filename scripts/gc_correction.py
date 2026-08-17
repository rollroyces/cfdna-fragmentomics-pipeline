#!/usr/bin/env python3
"""GC-content bias correction (LOESS regression over genomic bins).

GC bias: fragments from GC-poor or GC-rich regions are captured/amplified
less efficiently, so their coverage is systematically lower than fragments
from GC-neutral regions.  This is a sequencing artifact, not biology — it
must be removed before classification.

Method (standard LOESS approach, e.g. from cfDNA fragmentation pipelines):
  1. For each 100 kb bin: compute mean GC fraction from the reference.
  2. Compute the bin's observed fragment count (from DELFI extraction).
  3. Fit a LOESS curve: observed_count ~ f(GC).
  4. Correct each bin: corrected = observed / (f(GC) / mean(f)).
  5. Recompute the short/long ratio using corrected counts.

Requires scikit-learn (LOESS via local weighted regression is implemented
directly with numpy; no extra dependency beyond numpy/scipy).

Usage:
  python gc_correction.py --bins data/features/S1.delfi_100kb_ratio.npy \
      --counts data/features/S1.delfi_100kb_counts.npy \
      --gc data/references/hg38_100kb_gc.npy \
      --sample S1 --out-dir data/features

Output: features/<sample>.gc_corrected.npy  (corrected short/long ratio vector)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d


def loess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Local weighted linear regression (tricube kernel), scipy-free."""
    n = len(x)
    yhat = np.zeros(n)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    nbr = max(int(n * frac), 10)
    for i in range(n):
        lo = max(0, i - nbr // 2)
        hi = min(n, i + nbr // 2 + 1)
        xs_win, ys_win = xs[lo:hi], ys[lo:hi]
        d = np.abs(xs_win - xs[i])
        dmax = d.max()
        if dmax <= 0:
            yhat[i] = ys_win.mean()
            continue
        w = (1 - (d / dmax) ** 3) ** 3  # tricube
        wsum = w.sum()
        if wsum <= 0:
            yhat[i] = ys_win.mean()
            continue
        xbar = (w * xs_win).sum() / wsum
        ybar = (w * ys_win).sum() / wsum
        b = (w * (xs_win - xbar) * (ys_win - ybar)).sum() / \
            max((w * (xs_win - xbar) ** 2).sum(), 1e-12)
        yhat[i] = ybar + b * (xs[i] - xbar)
    # back to original order
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    return yhat[inv]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", required=True, help="per-bin short/long ratio .npy")
    ap.add_argument("--counts", required=True, help="per-bin fragment counts .npy")
    ap.add_argument("--gc", required=True, help="per-bin GC fraction .npy")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--loess-frac", type=float, default=0.15)
    args = ap.parse_args()

    ratio = np.load(args.bins)
    counts = np.load(args.counts)
    gc = np.load(args.gc)
    if not (len(ratio) == len(counts) == len(gc)):
        print(f"ERROR: length mismatch ratio={len(ratio)} counts={len(counts)} "
              f"gc={len(gc)}", file=sys.stderr)
        sys.exit(1)

    # Fit LOESS on bins with reasonable coverage
    mask = (counts > 0) & np.isfinite(gc)
    if mask.sum() < 50:
        print("ERROR: too few valid bins for GC correction", file=sys.stderr)
        sys.exit(1)

    fit = loess_smooth(gc[mask], counts[mask], frac=args.loess_frac)
    mean_fit = float(np.mean(fit))
    # correction factor per bin: counts / (fit/mean)  ->  ratio corrected
    correction = np.ones(len(counts))
    correction[mask] = mean_fit / np.maximum(fit, 1e-9)
    # apply to ratio: corrected = ratio * (counts before) / (counts after)
    #   counts_after = counts * correction  →  ratio_after ≈ ratio / correction
    corrected = np.ones(len(ratio))
    corrected[mask] = ratio[mask] / np.maximum(correction[mask], 1e-9)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.sample}.gc_corrected.npy")
    np.save(out, corrected)

    gc_corr = np.corrcoef(gc[mask], counts[mask])[0, 1]
    print(f"GC-correct [{args.sample}]: bins={mask.sum()} "
          f"raw GC~count corr={gc_corr:+.3f} -> {out}")
    with open(os.path.join(args.out_dir, f"{args.sample}.gc_meta.json"), "w") as f:
        json.dump({"sample": args.sample, "n_bins": int(mask.sum()),
                   "gc_count_corr": float(gc_corr),
                   "loess_frac": args.loess_frac}, f)


if __name__ == "__main__":
    main()
