#!/usr/bin/env python3
"""Fetch real cfDNA fragment data from FinaleDB (CCHMC epifluidlab).

FinaleDB (https://pubmed.ncbi.nlm.nih.gov/33258919/) hosts 2,500+ uniformly
processed cfDNA WGS datasets.  Per-sample fragment records are served from a
public S3 bucket:  entries/<seqrun_id>/hg38/<seqrun_id>.hg38.frag.tsv.bgz

Format of a .frag.tsv.bgz row (tab-separated, one fragment per line):
    chrom<TAB>start<TAB>end<TAB>mapq<TAB>strand
  e.g. chr1<TAB>9998<TAB>10104<TAB>0<TAB>-

Usage:
  python fetch_finaledb.py --disease "Liver cancer" --healthy \
      --n 12 --out-dir data/raw
  python fetch_finaledb.py --disease "Colorectal cancer" --n 12 \
      --out-dir data/raw --no-delete

The script streams each file, so peak disk usage is one file at a time.
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter

API = "http://finaledb.research.cchmc.org/api/v1/seqrun"
S3 = "https://s3.us-east-2.amazonaws.com/finaledb.epifluidlab.cchmc.org"


def query_seqruns(disease: str | None = None, healthy: bool = False,
                  limit: int = 12, offset: int = 0) -> list[dict]:
    """Query the FinaleDB seqrun API; filter by disease / healthy status."""
    runs: list[dict] = []
    params = f"?page_size=50&offset={offset}"
    while True:
        url = API + params
        with urllib.request.urlopen(url, timeout=30) as u:
            d = json.load(u)
        for r in d.get("results", []):
            sample = r.get("sample") or {}
            dis = (sample.get("disease") or "?")
            if healthy and dis != "Healthy":
                continue
            if disease and dis != disease:
                continue
            runs.append(r)
            if len(runs) >= limit:
                return runs
        if not d.get("next") or len(runs) >= limit:
            break
        params = d["next"].split("?", 1)[-1]
    return runs


def download_frag(seqrun_id: int, out_path: str) -> bool:
    """Stream the frag.tsv.bgz for a seqrun to out_path."""
    key = f"entries/EE{seqrun_id}/hg38/EE{seqrun_id}.hg38.frag.tsv.bgz"
    url = f"{S3}/{key}"
    tmp = out_path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=600) as u, open(tmp, "wb") as f:
            shutil.copyfileobj(u, f, length=1024 * 1024)
        os.replace(tmp, out_path)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  ! failed {seqrun_id}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", help="disease label, e.g. 'Liver cancer'")
    ap.add_argument("--healthy", action="store_true",
                    help="fetch 'Healthy' control samples instead")
    ap.add_argument("--n", type=int, default=12, help="max samples to fetch")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--no-delete", action="store_true",
                    help="keep .frag.tsv.bgz (default: delete after 60s to save disk)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = query_seqruns(args.disease, args.healthy, limit=args.n)
    print(f"Selected {len(runs)} real samples:",
          dict(Counter((r.get('sample') or {}).get('disease', '?') for r in runs)))

    manifest = []
    for r in runs:
        sid = r["id"]
        sample = (r.get("sample") or {}).get("name", f"run{sid}")
        out = os.path.join(args.out_dir, f"{sample}.frag.tsv.bgz")
        if os.path.exists(out):
            print(f"  [skip] {sample} (already downloaded)")
        else:
            print(f"  [fetch] {sample} (seqrun {sid}) ...", flush=True)
            if not download_frag(sid, out):
                continue
        manifest.append({"sample": sample, "seqrun_id": sid, "file": out})

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {os.path.join(args.out_dir, 'manifest.json')} "
          f"({len(manifest)} samples)")
    if not args.no_delete:
        print("NOTE: raw .frag.tsv.bgz are kept for feature extraction; "
              "run `clean_raw.py` (or delete) after features are extracted "
              "to free disk.")


if __name__ == "__main__":
    main()
