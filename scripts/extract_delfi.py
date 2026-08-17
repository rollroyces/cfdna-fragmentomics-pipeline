#!/usr/bin/env python3
"""Extract DELFI-style regional fragmentation features + Window Protection Score.

Two signals from the guideline's "big three":

1. DELFI profiling — bin the genome into 100 kb windows (and 5 Mb for
   chromosome-scale), compute the ratio of short fragments (100-150 bp)
   to long fragments (150-220 bp) in each window.  This reveals copy
   number aberrations and chromatin accessibility: high-expression,
   nucleosome-depleted loci in cancer cells fragment differently than
   silent regions → an epigenetic fingerprint.

2. Window Protection Score (WPS) — for each position, score how many
   fragments fully cover the position vs how many start/end near it.
   High WPS = nucleosome-protected; low WPS = open chromatin.
   (This implementation computes mean per-window WPS over the DELFI bins.)

Input  : FinaleDB .frag.tsv.bgz (chrom start end mapq strand)
Output : features/<sample>.delfi.json
         {
           "bins_100kb": { "chr1:1-100000": {"short": 12, "long": 40, "ratio": 0.30}, ... },
           "bins_5mb":   { "chr1:1-5000000": {"short": 600, "long": 2100, "ratio": 0.286}, ... },
           "mean_short_frac": f, "mean_long_frac": f,
           "genome_short_long_ratio": r
         }
  (bins_100kb is written to a companion .npy for speed; only aggregates in JSON)

Usage:
  python extract_delfi.py --input data/raw/S1.frag.tsv.bgz --sample S1 \
      --out-dir data/features
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

# hg38 chromosome sizes (canonical chromosomes 1-22, X, Y)
CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}
BIN_100KB = 100_000
BIN_5MB = 5_000_000


def _bin_map(chrom: str, bin_size: int) -> list[tuple[int, int]]:
    """chrom -> [(start, end), ...] windows covering the chromosome."""
    n = (CHROM_SIZES[chrom] + bin_size - 1) // bin_size
    return [(i * bin_size, min((i + 1) * bin_size, CHROM_SIZES[chrom]))
            for i in range(n)]


def extract(path: str, out_dir: str, sample: str) -> dict:
    # Pre-build window indices: chrom -> (bin_size -> array of starts)
    windows_100kb: dict[str, list[tuple[int, int]]] = {}
    windows_5mb: dict[str, list[tuple[int, int]]] = {}
    for chrom in CHROM_SIZES:
        windows_100kb[chrom] = _bin_map(chrom, BIN_100KB)
        windows_5mb[chrom] = _bin_map(chrom, BIN_5MB)

    # Counters: (chrom, bin_idx) -> [short, long] (separate arrays!)
    from collections import defaultdict
    short_100: dict[str, np.ndarray] = {c: np.zeros(len(windows_100kb[c]), dtype=np.int64)
                                        for c in CHROM_SIZES}
    long_100: dict[str, np.ndarray] = {c: np.zeros(len(windows_100kb[c]), dtype=np.int64)
                                       for c in CHROM_SIZES}
    short_5: dict[str, np.ndarray] = {c: np.zeros(len(windows_5mb[c]), dtype=np.int64)
                                      for c in CHROM_SIZES}
    long_5: dict[str, np.ndarray] = {c: np.zeros(len(windows_5mb[c]), dtype=np.int64)
                                     for c in CHROM_SIZES}
    # WPS: accumulate per-position (start/end events and coverage) per 100kb bin
    wps_support: dict[str, np.ndarray] = {c: np.zeros(len(windows_100kb[c]), dtype=np.float64)
                                          for c in CHROM_SIZES}
    wps_count: dict[str, np.ndarray] = {c: np.zeros(len(windows_100kb[c]), dtype=np.float64)
                                        for c in CHROM_SIZES}

    total_short = 0
    total_long = 0

    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if chrom not in CHROM_SIZES:
                continue
            flen = end - start
            if not (100 <= flen <= 220):
                continue
            # classify short (100-150) vs long (150-220)
            is_short = flen <= 150
            if is_short:
                total_short += 1
            else:
                total_long += 1
            # 100kb bin
            bi = start // BIN_100KB
            if bi < len(short_100[chrom]):
                if is_short:
                    short_100[chrom][bi] += 1
                else:
                    long_100[chrom][bi] += 1
            # 5Mb bin
            bi5 = start // BIN_5MB
            if bi5 < len(short_5[chrom]):
                if is_short:
                    short_5[chrom][bi5] += 1
                else:
                    long_5[chrom][bi5] += 1
            # WPS contribution: support = (# fully covering - # starting/ending here)
            # Approximate per-window: full-cover span contributes +span/win,
            # start/end events contribute -1.
            if bi < len(wps_support[chrom]):
                wps_support[chrom][bi] += (min(end, windows_100kb[chrom][bi][1])
                                           - max(start, windows_100kb[chrom][bi][0])) / BIN_100KB
                wps_count[chrom][bi] += 1.0
            # start/end events (both fragment endpoints) reduce WPS
            if bi < len(wps_support[chrom]):
                wps_support[chrom][bi] -= 2.0 / BIN_100KB

    # Convert counters to (short, long, ratio)
    bins_100kb = {}
    for c in CHROM_SIZES:
        for i, (s, e) in enumerate(windows_100kb[c]):
            short = int(short_100[c][i])
            long_ = int(long_100[c][i])
            bins_100kb[f"{c}:{s}-{e}"] = {
                "short": short, "long": long_,
                "ratio": float(short / max(long_, 1)),
            }

    bins_5mb = {}
    for c in CHROM_SIZES:
        for i, (s, e) in enumerate(windows_5mb[c]):
            short = int(short_5[c][i])
            long_ = int(long_5[c][i])
            bins_5mb[f"{c}:{s}-{e}"] = {
                "short": short, "long": long_,
                "ratio": float(short / max(long_, 1)),
            }

    ratios = np.array([b["ratio"] for b in bins_100kb.values()])
    wps_vals = np.array([float(wps_support[c].sum()) for c in CHROM_SIZES])

    result = {
        "sample": sample,
        "genome_short_long_ratio": float(total_short / max(total_long, 1)),
        "mean_short_frac": float(total_short / max(total_short + total_long, 1)),
        "mean_window_ratio_100kb": float(np.mean(ratios)),
        "median_window_ratio_100kb": float(np.median(ratios)),
        "p10_window_ratio_100kb": float(np.percentile(ratios, 10)),
        "p90_window_ratio_100kb": float(np.percentile(ratios, 90)),
        "bins_5mb": bins_5mb,
        "n_windows_100kb": len(ratios),
    }
    # Save per-bin fragment counts for GC correction
    counts_100kb = np.array([int(short_100[c][i]) + int(long_100[c][i])
                             for c in CHROM_SIZES
                             for i in range(len(windows_100kb[c]))])
    np.save(os.path.join(out_dir, f"{sample}.delfi_100kb_counts.npy"), counts_100kb)
    np.save(os.path.join(out_dir, f"{sample}.delfi_100kb_ratio.npy"), ratios)
    np.save(os.path.join(out_dir, f"{sample}.wps_100kb.npy"), wps_vals)

    # 5Mb ratio vector + CNV coverage profile (median-normalized)
    ratio_5mb = np.array([float(b["ratio"]) for c in CHROM_SIZES
                          for b in [bins_5mb[f"{c}:{s}-{e}"]
                                    for (s, e) in windows_5mb[c]]])
    cover_5mb = np.array([int(short_5[c][i]) + int(long_5[c][i])
                          for c in CHROM_SIZES
                          for i in range(len(windows_5mb[c]))])
    # median-normalize coverage to 1.0 → depth-independent CNV signal
    med = float(np.median(cover_5mb[cover_5mb > 0]))
    cover_norm = cover_5mb.astype(float) / max(med, 1.0)
    np.save(os.path.join(out_dir, f"{sample}.delfi_5mb_ratio.npy"), ratio_5mb)
    np.save(os.path.join(out_dir, f"{sample}.delfi_5mb_coverage.npy"), cover_norm)

    out = os.path.join(out_dir, f"{sample}.delfi.json")
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"DELFI [{sample}]: short={total_short:,} long={total_long:,} "
          f"S/L={result['genome_short_long_ratio']:.3f} "
          f"mean-win-ratio={result['mean_window_ratio_100kb']:.3f}  ->  {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", default="data/features")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    extract(args.input, args.out_dir, args.sample)


if __name__ == "__main__":
    main()
