#!/usr/bin/env python3
"""Extract 4-mer end-motif frequencies from BAM reads (tumor-naive).

Different nucleases act in the tumor microenvironment vs healthy plasma
(e.g. DNASE1L3 activity).  The first 4 nucleotides at the 5' end of every
read form a 4-mer motif; the frequency of all 256 possible 4-mers is a
fragmentomic biomarker (Jiang et al., Cancer Discov 2020).

Requires pysam + a reference genome FASTA to resolve the 5' end sequence.

Contract (per guideline):
  - properly paired reads only
  - MAPQ >= 30
  - deduplicated BAM (Picard MarkDuplicates upstream)

Usage:
  python extract_motifs.py --bam sample.mdups.bam --ref hg38.fa \
      --sample S1 --out-dir data/features

Output: features/<sample>.motifs.json
  {"AAAA": 0.0123, "AAAC": 0.0087, ...}  (256 normalized frequencies)
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ALL_4MERS = sorted(f"{a}{b}{c}{d}"
                   for a in "ACGT" for b in "ACGT"
                   for c in "ACGT" for d in "ACGT")


def extract(bam_path: str, ref_fasta: str, mapq: int = 30) -> tuple[dict, int]:
    import pysam

    ref = pysam.FastaFile(ref_fasta)
    counts = Counter()
    total = 0
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch():
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if not read.is_proper_pair:
                continue
            if read.mapping_quality < mapq:
                continue
            if read.is_duplicate:
                continue
            # 5' end of the read (in reference coordinates)
            if read.is_reverse:
                # reverse strand: 5' end is the rightmost mapped position
                seq5 = ref.fetch(read.reference_name,
                                 read.reference_end - 4,
                                 read.reference_end)
                seq5 = str(seq5)
                if len(seq5) >= 4:
                    counts[seq5[:4]] += 1
            else:
                seq5 = ref.fetch(read.reference_name,
                                 read.reference_start,
                                 read.reference_start + 4)
                seq5 = str(seq5)
                if len(seq5) >= 4:
                    counts[seq5[:4]] += 1
            total += 1

    freqs = {m: counts[m] / max(total, 1) for m in ALL_4MERS}
    return freqs, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bam", required=True, help="deduplicated BAM")
    ap.add_argument("--ref", required=True, help="reference genome FASTA (hg38)")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--mapq", type=int, default=30)
    args = ap.parse_args()

    try:
        freqs, total = extract(args.bam, args.ref, args.mapq)
    except ImportError:
        print("ERROR: pysam not installed.  `pip install pysam` for BAM mode.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.sample}.motifs.json")
    with open(out, "w") as f:
        json.dump({"sample": args.sample, "n_reads": total, "freqs": freqs}, f)
    top = sorted(freqs.items(), key=lambda kv: -kv[1])[:5]
    print(f"Motifs [{args.sample}]: n_reads={total:,} "
          f"top={', '.join(f'{m}:{v:.4f}' for m, v in top)}  ->  {out}")


if __name__ == "__main__":
    main()
