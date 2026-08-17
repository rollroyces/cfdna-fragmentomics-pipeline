#!/usr/bin/env python3
"""Build the hg38 per-100kb-bin GC fraction reference (no reference FASTA needed).

Uses the known hg38 GC content distribution.  For exact per-bin GC, download
hg38.fa and run:  `python build_gc_reference.py --fasta hg38.fa`.
The built-in approximation uses the canonical chromosome GC fractions.

Output: data/references/hg38_100kb_gc.npy (one value per 100kb bin,
ordered chr1..chr22,X,Y — same order as extract_delfi.py).

Usage:
  python build_gc_reference.py [--fasta hg38.fa] [--out data/references/hg38_100kb_gc.npy]
"""
import argparse
import os
import sys

import numpy as np

# Canonical hg38 chromosome-level GC fractions (per-100kb bins vary ±8%).
CHROM_GC = {
    "chr1": 0.418, "chr2": 0.405, "chr3": 0.398, "chr4": 0.381, "chr5": 0.394,
    "chr6": 0.401, "chr7": 0.410, "chr8": 0.389, "chr9": 0.413, "chr10": 0.414,
    "chr11": 0.416, "chr12": 0.415, "chr13": 0.356, "chr14": 0.379, "chr15": 0.405,
    "chr16": 0.441, "chr17": 0.458, "chr18": 0.384, "chr19": 0.486, "chr20": 0.438,
    "chr21": 0.375, "chr22": 0.466, "chrX": 0.394, "chrY": 0.378,
}
CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}
BIN = 100_000


def gc_from_fasta(fasta: str) -> np.ndarray:
    """Exact per-100kb GC from a reference FASTA (slower but exact)."""
    values = []
    seq = []
    chrom = None
    import gzip
    def flush():
        nonlocal seq
        if seq and chrom:
            s = "".join(seq).upper()
            for i in range(0, len(s), BIN):
                chunk = s[i:i + BIN]
                gc = (chunk.count("G") + chunk.count("C")) / max(len(chunk), 1)
                values.append(gc)
        seq = []
    opener = gzip.open if fasta.endswith(".gz") else open
    with opener(fasta, "rt") as f:
        for line in f:
            if line.startswith(">"):
                flush()
                chrom = line[1:].strip().split()[0]
            elif chrom in CHROM_SIZES:
                seq.append(line.strip())
    flush()
    return np.asarray(values)


def gc_approx() -> np.ndarray:
    """Chromosome-level approximation (exact when a FASTA is unavailable)."""
    values = []
    for chrom in CHROM_SIZES:
        n = (CHROM_SIZES[chrom] + BIN - 1) // BIN
        values.extend([CHROM_GC[chrom]] * n)
    return np.asarray(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", help="hg38.fa (exact) — default: chromosome-level approx")
    ap.add_argument("--out", default="data/references/hg38_100kb_gc.npy")
    args = ap.parse_args()
    gc = gc_from_fasta(args.fasta) if args.fasta else gc_approx()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, gc)
    print(f"GC reference: {len(gc)} bins ({'exact FASTA' if args.fasta else 'approx'}) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
