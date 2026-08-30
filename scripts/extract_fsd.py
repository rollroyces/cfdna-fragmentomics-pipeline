#!/usr/bin/env python3
"""Extract the Fragment Size Distribution (FSD) from cfDNA fragment data.

cfDNA is not cleaved randomly — it wraps around nucleosomes.  Healthy plasma
cfDNA peaks at ~167 bp (nucleosome + linker).  Tumor-derived cfDNA is
statistically shorter, peaking closer to ~145 bp (nucleosome-disrupted,
less-protected chromatin).

Input  : FinaleDB .frag.tsv.bgz  (chrom start end mapq strand per fragment)
         OR a BAM file (mode=bam, uses TLEN of properly-paired reads,
         MAPQ >= 30, the guideline's BAM-mode contract).
Output : features/<sample>.fsd.json
         {
           "fragment_count": N,
           "median_length", "mean_length", "mode_length",
           "p10", "p25", "p75", "p90",
           "short_fraction_100_150": f,   # tumor-enriched window
           "long_fraction_150_220":  f,   # healthy window
           "short_long_ratio": r,          # DELFI-style ratio
           "size_bins": { "140-145": 0.012, ... }   # 5 bp bins, normalized
         }

Usage:
  python extract_fsd.py --input data/raw/S1.frag.tsv.bgz \
      --sample S1 --out-dir data/features
  python extract_fsd.py --input sample.bam --mode bam --sample S1 \
      --out-dir data/features
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nuc_features import FSD_BIN_START, FSD_BIN_END, FSD_BIN_STRIDE  # noqa


def extract_from_frag_tsv(path: str, mapq_threshold: int = 0) -> np.ndarray:
    """Read fragment lengths from FinaleDB frag.tsv.bgz (length = end - start).

    FinaleDB's pre-computed fragment records follow the bedtools bamtobed
    -bedpe schema: `chrom start end name mapq strand` (6 columns). The
    pipeline's older 5-column format `[chrom, start, end, mapq, strand]`
    silently dropped every real fragment; the fixed index 4 = mapq below.
    """
    lengths = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                start, end, mapq = int(parts[1]), int(parts[2]), int(parts[4])
            except ValueError:
                continue
            if mapq < mapq_threshold:
                continue
            frag_len = end - start
            if FSD_BIN_START <= frag_len <= FSD_BIN_END:
                lengths.append(frag_len)
    return np.asarray(lengths, dtype=np.int32)


def extract_from_bam(path: str, mapq_threshold: int = 30) -> np.ndarray:
    """Read TLEN of properly-paired reads from a BAM (MAPQ >= 30).

    Requires pysam.  Follows the guideline's BAM-mode contract:
    strictly proper pairs, MAPQ >= 30, absolute template length.
    """
    import pysam
    lengths = []
    with pysam.AlignmentFile(path, "rb") as bam:
        for read in bam.fetch():
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if not read.is_proper_pair:
                continue
            if read.mapping_quality < mapq_threshold:
                continue
            tlen = abs(read.template_length)
            if FSD_BIN_START <= tlen <= FSD_BIN_END:
                lengths.append(tlen)
    return np.asarray(lengths, dtype=np.int32)


def summarize(lengths: np.ndarray) -> dict:
    n = len(lengths)
    if n == 0:
        raise ValueError("no fragments in input")
    hist, edges = np.histogram(lengths,
                               bins=range(FSD_BIN_START, FSD_BIN_END + 1, FSD_BIN_STRIDE))
    bin_labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
    size_bins = {bin_labels[i]: float(hist[i] / n) for i in range(len(hist))}
    short = int(((lengths >= 100) & (lengths <= 150)).sum())
    long_ = int(((lengths >= 150) & (lengths <= 220)).sum())
    return {
        "fragment_count": int(n),
        "median_length": float(np.median(lengths)),
        "mean_length": float(np.mean(lengths)),
        "mode_length": float(edges[int(np.argmax(hist))] + 2.5),
        "p10": float(np.percentile(lengths, 10)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "short_fraction_100_150": float(short / n),
        "long_fraction_150_220": float(long_ / n),
        "short_long_ratio": float(short / max(long_, 1)),
        "size_bins": size_bins,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", choices=["frag", "bam"], default="frag")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--mapq", type=int, default=0,
                    help="MAPQ threshold (frag mode default 0; bam mode default 30)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mapq = args.mapq if args.mode == "bam" else max(args.mapq, 0)
    if args.mode == "bam":
        lengths = extract_from_bam(args.input, mapq_threshold=mapq)
    else:
        lengths = extract_from_frag_tsv(args.input, mapq_threshold=mapq)

    summary = summarize(lengths)
    out = os.path.join(args.out_dir, f"{args.sample}.fsd.json")
    with open(out, "w") as f:
        json.dump({"sample": args.sample, "mode": args.mode, **summary}, f, indent=2)
    print(f"FSD [{args.sample}]: n={summary['fragment_count']:,} "
          f"median={summary['median_length']:.1f}bp "
          f"mode={summary['mode_length']:.1f}bp "
          f"short(100-150)={summary['short_fraction_100_150']:.4f} "
          f"short/long={summary['short_long_ratio']:.3f}  ->  {out}")


if __name__ == "__main__":
    main()
