#!/usr/bin/env python3
"""Re-extract a subset with the NEW features (mean-length + 4-mer motifs).

Download → FSD + DELFI (now incl. mean-length) + motifs → delete raw.
Used to test whether the new features improve AUC before committing to a
full 627-sample re-download.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO, "scripts")
RAW = os.path.join(REPO, "data", "raw")
FEAT = os.path.join(REPO, "data", "features")
REF = os.path.join(REPO, "data", "references", "hg38.2bit")

sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)
from run_real_cohort import process_sample, PY  # noqa: E402


def reextract(sample, seqrun_id):
    """download + delfi(mean-length) + motifs, then delete raw."""
    raw = os.path.join(RAW, f"{sample}.frag.tsv.bgz")
    if not os.path.exists(raw):
        from fetch_finaledb import download_frag
        if not download_frag(seqrun_id, raw, max_mb=500):
            return False
    # DELFI (now writes mean-length) — skip FSD (already have it, unchanged)
    from subprocess import run as srun
    srun([PY, os.path.join(SCRIPTS, "extract_delfi.py"), "--input", raw,
          "--sample", sample, "--out-dir", FEAT], check=True)
    # motifs
    srun([PY, os.path.join(SCRIPTS, "extract_motifs_frag.py"), "--input", raw,
          "--ref", REF, "--sample", sample, "--out-dir", FEAT], check=True)
    os.remove(raw)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(FEAT, "labels_cross_study.tsv"))
    ap.add_argument("--n-cancer", type=int, default=50)
    ap.add_argument("--n-healthy", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    # map sample -> seqrun_id (query FinaleDB once)
    sys.path.insert(0, SCRIPTS)
    from fetch_finaledb import query_seqruns
    id_map = {}
    for pub in (6, 8):
        for r in query_seqruns(None, healthy=True, limit=1000, publication=pub):
            s = (r.get("sample") or {}).get("name")
            id_map[s] = r["id"]
        for dis in ("Liver cancer", "Lung cancer", "Breast cancer",
                    "Pancreatic cancer", "Ovarian cancer", "Colorectal cancer",
                    "Gastric cancer", "Bile duct cancer", "Duodenal cancer"):
            for r in query_seqruns(dis, limit=1000, publication=pub):
                s = (r.get("sample") or {}).get("name")
                id_map[s] = r["id"]

    cancer, healthy = [], []
    for line in open(args.labels):
        p = line.strip().split("\t")
        if p[1] == "cancer" and len(cancer) < args.n_cancer:
            cancer.append(p[0])
        elif p[1] == "healthy" and len(healthy) < args.n_healthy:
            healthy.append(p[0])
    samples = cancer + healthy
    print(f"Re-extracting {len(samples)} samples ({len(cancer)} cancer + "
          f"{len(healthy)} healthy) ...")

    work = [(s, id_map[s]) for s in samples if s in id_map]
    ok = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(reextract, s, sid) for s, sid in work]
        for i, fut in enumerate(as_completed(futs), 1):
            ok += int(fut.result())
            if i % 10 == 0:
                print(f"  [{i}/{len(work)}] done", flush=True)
    print(f"done: {ok}/{len(work)}")


if __name__ == "__main__":
    main()
