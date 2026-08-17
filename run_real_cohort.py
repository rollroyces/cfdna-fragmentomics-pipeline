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
import re
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from fetch_finaledb import query_seqruns, download_frag  # noqa: E402

# Cell-line / technical-control names (not patient plasma) — exclude.
# e.g. GM1100 (B-lymphocyte line) is labeled "Liver cancer" in FinaleDB
# but is a reference sample; its fragmentation differs from in-vivo cfDNA.
CELL_LINE_PATTERN = re.compile(
    r'^(GM\d+|HeLa|HepG2|K562|HL60|Jurkat|Raji|MCF7|U937|THP1|HEK293|'
    r'HCT116|SW480|A549|GM12878)', re.I)


def is_cell_line(sample: str) -> bool:
    return bool(CELL_LINE_PATTERN.match(sample))

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


def process_sample(sample: str, seqrun_id: int, keep_raw: bool = False,
                   max_mb: float = 500):
    raw = os.path.join(RAW, f"{sample}.frag.tsv.bgz")
    if not os.path.exists(raw):
        print(f"  [fetch] {sample} (seqrun {seqrun_id}) ...", flush=True)
        if not download_frag(seqrun_id, raw, max_mb=max_mb):
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
    ap.add_argument("--parallel", type=int, default=1,
                    help="concurrent sample downloads (I/O-bound; default 1)")
    ap.add_argument("--publication", type=int, default=None,
                    help="restrict to one FinaleDB study (6 = Jiang 2015 low-pass)")
    ap.add_argument("--max-mb", type=float, default=500,
                    help="reject frag files larger than this (deep-WGS guard)")
    ap.add_argument("--pca", action="store_true", help="DELFI-style full-profile PCA")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(FEAT, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # 1. Discover samples
    cancer_runs = query_seqruns(args.cancer, healthy=False, limit=args.n_cancer,
                                publication=args.publication)
    healthy_runs = query_seqruns(None, healthy=args.healthy, limit=args.n_healthy,
                                 publication=args.publication) \
        if args.healthy else []
    labels = {}
    print(f"Cohort: {len(cancer_runs)} cancer + {len(healthy_runs)} healthy "
          f"(publication={args.publication}, max {args.max_mb:.0f}MB)")

    # 2. Stream: fetch → extract → delete (skip samples fully processed)
    processed = []
    def _done(s):
        return os.path.exists(os.path.join(FEAT, f"{s}.fsd.json")) and \
               os.path.exists(os.path.join(FEAT, f"{s}.delfi_5mb_ratio.npy"))

    work = []  # (sample, seqrun_id, label)
    for r in cancer_runs:
        s = (r.get("sample") or {}).get("name", f"run{r['id']}")
        if is_cell_line(s):
            print(f"  [exclude] {s} (cell-line control, not patient plasma)")
            continue
        if _done(s):
            processed.append(s); labels[s] = "cancer"; continue
        work.append((s, r["id"], "cancer"))
    for r in healthy_runs:
        s = (r.get("sample") or {}).get("name", f"run{r['id']}")
        if is_cell_line(s):
            print(f"  [exclude] {s} (cell-line control, not patient plasma)")
            continue
        if _done(s):
            processed.append(s); labels[s] = "healthy"; continue
        work.append((s, r["id"], "healthy"))

    if args.parallel > 1 and work:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Fetching {len(work)} samples with {args.parallel} workers ...")
        def _proc(item):
            s, sid, lab = item
            ok = process_sample(s, sid, args.keep_raw, max_mb=args.max_mb)
            return s, lab, ok
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_proc, w): w for w in work}
            done = 0
            for fut in as_completed(futs):
                s, lab, ok = fut.result()
                done += 1
                if ok:
                    labels[s] = lab
                    processed.append(s)
                if done % 5 == 0:
                    print(f"  [{done}/{len(work)}] done", flush=True)
    else:
        for s, sid, lab in work:
            if process_sample(s, sid, args.keep_raw, max_mb=args.max_mb):
                labels[s] = lab
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
    cmd = [PY, os.path.join(SCRIPTS, "train_classifier.py"),
           "--features", FEAT, "--labels", labels_path, "--out", args.out,
           "--model", args.model, "--cv", str(args.cv)]
    if args.pca:
        cmd.append("--pca")
    run(*cmd)
    res = json.load(open(os.path.join(args.out, "classifier_results.json")))
    print("\n=== REAL-DATA FRAGMENTOMICS CLASSIFICATION ===")
    print(f"  n={res['n_samples']}  AUC={res['auc_mean']:.3f}±{res['auc_std']:.3f}  "
          f"Sens@95%={res['sens95_mean']:.3f}  Sens@99%={res['sens99_mean']:.3f}")


if __name__ == "__main__":
    main()
