#!/usr/bin/env python3
"""Cross-study pan-cancer cohort: Jiang 2015 (pub 6) + Cristiano 2019 (pub 8).

Both cancer and healthy classes span BOTH studies, so study batch effects
partially cancel. Study-aware z-score harmonization is applied inside each
CV fold (never leaks test-set statistics).

Cancer: Jiang HCC + Cristiano lung/breast/pancreatic/ovarian/colorectal/
        gastric/bile-duct/duodenal.
Healthy: Jiang healthy + Cristiano healthy.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from fetch_finaledb import query_seqruns  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO, "scripts")
RAW = os.path.join(REPO, "data", "raw")
FEAT = os.path.join(REPO, "data", "features")

# import the runner's helpers
sys.path.insert(0, REPO)
from run_real_cohort import process_sample, is_cell_line  # noqa: E402

CRISTIANO_CANCERS = ["Lung cancer", "Breast cancer", "Pancreatic cancer",
                     "Ovarian cancer", "Colorectal cancer", "Gastric cancer",
                     "Bile duct cancer", "Duodenal cancer"]


def discover():
    """Return (cancer_runs, healthy_runs) with study tags, patient-only."""
    cancer, healthy = [], []
    # Jiang 2015 HCC (pub 6)
    for r in query_seqruns("Liver cancer", limit=500, publication=6):
        s = (r.get("sample") or {}).get("name")
        if is_cell_line(s or ""):
            continue
        cancer.append(("jiang", s, r["id"]))
    for r in query_seqruns(None, healthy=True, limit=500, publication=6):
        s = (r.get("sample") or {}).get("name")
        if is_cell_line(s or ""):
            continue
        healthy.append(("jiang", s, r["id"]))
    # Cristiano 2019 (pub 8)
    for dis in CRISTIANO_CANCERS:
        for r in query_seqruns(dis, limit=500, publication=8):
            s = (r.get("sample") or {}).get("name")
            if is_cell_line(s or ""):
                continue
            cancer.append(("cristiano", s, r["id"]))
    for r in query_seqruns(None, healthy=True, limit=500, publication=8):
        s = (r.get("sample") or {}).get("name")
        if is_cell_line(s or ""):
            continue
        healthy.append(("cristiano", s, r["id"]))
    return cancer, healthy


def fetch(items, max_mb, parallel):
    work = []
    seen = set()
    for study, s, sid in items:
        if s in seen:
            continue
        seen.add(s)
        if not (os.path.exists(os.path.join(FEAT, f"{s}.fsd.json")) and
                os.path.exists(os.path.join(FEAT, f"{s}.delfi_5mb_ratio.npy"))):
            work.append((study, s, sid))
    if not work:
        return
    print(f"Fetching {len(work)} samples ({parallel} workers) ...")
    def _proc(item):
        study, s, sid = item
        return s, process_sample(s, sid, keep_raw=False, max_mb=max_mb)
    ok = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = [ex.submit(_proc, w) for w in work]
        for i, fut in enumerate(as_completed(futs), 1):
            s, done = fut.result()
            ok += int(done)
            if i % 10 == 0:
                print(f"  [{i}/{len(work)}] done", flush=True)
    print(f"  fetched {ok}/{len(work)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-mb", type=float, default=500)
    ap.add_argument("--model", default="lr")
    ap.add_argument("--pca-n", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(FEAT, exist_ok=True)

    cancer, healthy = discover()
    print(f"Cross-study cohort: {len(cancer)} cancer + {len(healthy)} healthy")
    fetch(cancer + healthy, args.max_mb, args.parallel)

    # Write labels with study column (sample \t label \t study), deduped
    labels_path = os.path.join(FEAT, "labels_cross_study.tsv")
    written = set()
    with open(labels_path, "w") as f:
        for study, s, sid in cancer:
            if s in written:
                continue
            written.add(s)
            f.write(f"{s}\tcancer\t{study}\n")
        for study, s, sid in healthy:
            if s in written:
                continue
            written.add(s)
            f.write(f"{s}\thealthy\t{study}\n")
    print(f"Labels: {labels_path} ({len(cancer) + len(healthy)} samples)")

    from subprocess import run
    run([sys.executable, os.path.join(SCRIPTS, "train_classifier.py"),
         "--features", FEAT, "--labels", labels_path, "--out", args.out,
         "--model", args.model, "--cv", "5", "--pca", "--pca-n",
         str(args.pca_n), "--harmonize"], check=False)


if __name__ == "__main__":
    main()
