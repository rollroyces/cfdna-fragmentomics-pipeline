#!/usr/bin/env python3
"""Run the full real-data cohort: fetch → extract features → delete raw → classify.

Disk-friendly: streams ONE sample at a time (download ~170MB frag.tsv,
extract FSD+DELFI+WPS, delete the raw file) so peak disk usage stays
under ~400MB — viable on laptops.

The Snakemake workflow (main.smk) is the production orchestrator for
batch/cluster runs; this script is the single-machine streaming path.

Usage:
  python run_real_cohort.py --cancer "Liver cancer" --healthy \
      --n-cancer 6 --n-healthy 6 --out results

Output:
  data/features/<sample>.{fsd.json,delfi.json,*.npy,...}   (kept)
  results/classifier_results.json                          (AUC etc.)
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from fetch_finaledb import query_seqruns, download_frag  # noqa: E402

PY = sys.executable
SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
FEAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "features")


def run(*args):
    r = subprocess.run(list(args), capture_output=True, text=True)
    if r.returncode != 0:
        print("  ! stderr:", r.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def process_sample(sample: str, seqrun_id: int, keep_raw: bool = False):
    raw = os.path.join(RAW, f"{sample}.frag.tsv.bgz")
    if not os.path.exists(raw):
        print(f"  [fetch] {sample} (seqrun {seqrun_id}) ...", flush=True)
        if not download_frag(seqrun_id, raw):
            return False
    print(f"  [extract] {sample} ...", flush=True)
    run(PY, os.path.join(SCRIPTS, "extract_fsd.py"),
        "--input", raw, "--sample", sample, "--out-dir", FEAT)
    run(PY, os.path.join(SCRIPTS, "extract_delfi.py"),
        "--input", raw, "--sample", sample, "--out-dir", FEAT)
    if not keep_raw and raw.endswith(".frag.tsv.bgz"):
        os.remove(raw)
        print(f"  [clean] removed {os.path.basename(raw)}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cancer", default="Liver cancer")
    ap.add_argument("--healthy", action="store_true", help="also fetch healthy controls")
    ap.add_argument("--n-cancer", type=int, default=6)
    ap.add_argument("--n-healthy", type=int, default=6)
    ap.add_argument("--out", default="results")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(FEAT, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # 1. Discover samples
    cancer_runs = query_seqruns(args.cancer, healthy=False, limit=args.n_cancer)
    healthy_runs = query_seqruns(None, healthy=args.healthy, limit=args.n_healthy) \
        if args.healthy else []
    labels = {}
    print(f"Cohort: {len(cancer_runs)} cancer + {len(healthy_runs)} healthy")

    # 2. Stream: fetch → extract → delete (skip samples fully processed)
    processed = []
    def _done(s):
        return os.path.exists(os.path.join(FEAT, f"{s}.fsd.json")) and \
               os.path.exists(os.path.join(FEAT, f"{s}.delfi_5mb_ratio.npy"))
    for r in cancer_runs:
        s = (r.get("sample") or {}).get("name", f"run{r['id']}")
        if _done(s):
            processed.append(s); labels[s] = "cancer"; continue
        if process_sample(s, r["id"], args.keep_raw):
            labels[s] = "cancer"
            processed.append(s)
    for r in healthy_runs:
        s = (r.get("sample") or {}).get("name", f"run{r['id']}")
        if _done(s):
            processed.append(s); labels[s] = "healthy"; continue
        if process_sample(s, r["id"], args.keep_raw):
            labels[s] = "healthy"
            processed.append(s)
    if len(processed) < 4:
        print("ERROR: too few samples processed", file=sys.stderr)
        sys.exit(1)

    # 3. GC correction
    gc_ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "references", "hg38_100kb_gc.npy")
    for s in processed:
        bins = os.path.join(FEAT, f"{s}.delfi_100kb_ratio.npy")
        counts = os.path.join(FEAT, f"{s}.delfi_100kb_counts.npy")
        if os.path.exists(bins) and os.path.exists(counts):
            run(PY, os.path.join(SCRIPTS, "gc_correction.py"),
                "--bins", bins, "--counts", counts, "--gc", gc_ref,
                "--sample", s, "--out-dir", FEAT)

    # 4. Labels + classify
    labels_path = os.path.join(FEAT, "labels.tsv")
    with open(labels_path, "w") as f:
        for s, lab in sorted(labels.items()):
            f.write(f"{s}\t{lab}\n")
    print(f"\nLabels written: {labels_path} ({len(labels)} samples)")
    run(PY, os.path.join(SCRIPTS, "train_classifier.py"),
        "--features", FEAT, "--labels", labels_path, "--out", args.out,
        "--model", args.model, "--cv", str(args.cv))
    res = json.load(open(os.path.join(args.out, "classifier_results.json")))
    print("\n=== REAL-DATA FRAGMENTOMICS CLASSIFICATION ===")
    print(f"  n={res['n_samples']}  AUC={res['auc_mean']:.3f}±{res['auc_std']:.3f}  "
          f"Sens@95%={res['sens95_mean']:.3f}  Sens@99%={res['sens99_mean']:.3f}")


if __name__ == "__main__":
    main()
