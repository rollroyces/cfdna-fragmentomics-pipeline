#!/usr/bin/env python3
"""HONEST benchmark — no optimization, no cherry-picking.

- 5 different random seeds for CV split
- Report mean +/- std across seeds
- Test every claim: 5-channel, 8-channel, single-study, cross-study
- Naive cross-study as a negative control
- Single-study baseline vs re-extracted subset for fair ablation

CLI:
  python scripts/honest_benchmark.py [--features-dir DIR]

By default reads from data/features/. With --features-dir you can
point at a different cohort root (e.g. /tmp/cristiano_only/ for
single-study debugging).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
from train_classifier import _harmonize

FEAT = "data/features"
SEEDS = [42, 13, 7, 99, 1234]


def fsd_vec(s, feat_dir=FEAT):
    """196-bin fragment-length histogram from FSD JSON."""
    p = os.path.join(feat_dir, f"{s}.fsd.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        sb = json.load(f)["size_bins"]
    keys = sorted(sb, key=lambda k: int(k.split("-")[0]))
    return np.array([sb[k] for k in keys], dtype=float)


def load5(labels, studies, feat_dir=FEAT):
    """5-channel: 5mb_ratio + 5mb_coverage + 100kb_ratio + 100kb_counts + FSD."""
    rows, order, y = [], [], []
    for s in sorted(labels):
        r5 = os.path.join(feat_dir, f"{s}.delfi_5mb_ratio.npy")
        c5 = os.path.join(feat_dir, f"{s}.delfi_5mb_coverage.npy")
        r100 = os.path.join(feat_dir, f"{s}.delfi_100kb_ratio.npy")
        c100 = os.path.join(feat_dir, f"{s}.delfi_100kb_counts.npy")
        if not all(os.path.exists(p) for p in (r5, c5, r100, c100)):
            continue
        sb = fsd_vec(s, feat_dir)
        if sb is None:
            continue
        cn = np.load(c100) / np.median(np.load(c100))
        v = np.concatenate([np.load(r5), np.load(c5), np.load(r100), cn, sb])
        rows.append(v); order.append(s); y.append(labels[s])
    return (np.asarray(rows), np.asarray(y),
            np.array([studies.get(s, "6") for s in order]))


def load8(labels, studies, feat_dir=FEAT):
    """8-channel: 5 + 100kb_meanlen + 5mb_meanlen + motifs."""
    rows, order, y = [], [], []
    for s in sorted(labels):
        r5 = os.path.join(feat_dir, f"{s}.delfi_5mb_ratio.npy")
        c5 = os.path.join(feat_dir, f"{s}.delfi_5mb_coverage.npy")
        r100 = os.path.join(feat_dir, f"{s}.delfi_100kb_ratio.npy")
        c100 = os.path.join(feat_dir, f"{s}.delfi_100kb_counts.npy")
        ml100 = os.path.join(feat_dir, f"{s}.delfi_100kb_meanlen.npy")
        ml5 = os.path.join(feat_dir, f"{s}.delfi_5mb_meanlen.npy")
        mot = os.path.join(feat_dir, f"{s}.motifs.npy")
        if not all(os.path.exists(p) for p in (r5, c5, r100, c100, ml100, ml5, mot)):
            continue
        sb = fsd_vec(s, feat_dir)
        if sb is None:
            continue
        cn = np.load(c100) / np.median(np.load(c100))
        v = np.concatenate([np.load(r5), np.load(c5), np.load(r100), cn, sb,
                            np.load(ml100), np.load(ml5), np.load(mot)])
        rows.append(v); order.append(s); y.append(labels[s])
    return (np.asarray(rows), np.asarray(y),
            np.array([studies.get(s, "6") for s in order]))


def eval_cv(X, y, st, pca_n, harmonize, seeds):
    aucs, s95s, s99s = [], [], []
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            if harmonize:
                Xtr, sc = _harmonize(X[tr], st[tr], None)
                Xte, _ = _harmonize(X[te], st[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca)).fit(Xtr)
            m = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr), y[tr])
            ys.extend(list(m.predict_proba(pca.transform(Xte))[:, 1]))
            yt.extend(list(y[te]))
        aucs.append(roc_auc_score(yt, ys))
        fpr, tpr, _ = roc_curve(yt, ys)

        def sat(t):
            idx = np.where(fpr <= t)[0]
            return float(tpr[idx[-1]]) if len(idx) else 0.0
        s95s.append(sat(0.05)); s99s.append(sat(0.01))
    return (np.mean(aucs), np.std(aucs),
            np.mean(s95s), np.std(s95s),
            np.mean(s99s), np.std(s99s))


def run(name, X, y, st, pca_n, harmonize):
    a, sa, s95, ss95, s99, ss99 = eval_cv(X, y, st, pca_n, harmonize, SEEDS)
    print(f"{name:55s} N={len(y):3d}  AUC {a:.4f}±{sa:.4f}  "
          f"S95 {s95:.3f}±{ss95:.3f}  S99 {s99:.3f}±{ss99:.3f}")
    return a


def _load_labels(labels_path):
    """Load labels TSV into (labels_dict, studies_dict) using a `with` block."""
    labels, studies = {}, {}
    with open(labels_path) as f:
        for line in f:
            p = line.strip().split("\t")
            labels[p[0]] = 1 if p[1] == "cancer" else 0
            studies[p[0]] = p[2] if len(p) >= 3 else "unknown"
    return labels, studies


def run_honest_benchmark(feat_dir=FEAT):
    """Run all 5 sections (A through E) of the honest benchmark.

    The previous version of this script had all work at module level,
    which meant `python scripts/honest_benchmark.py --help` triggered
    the entire 5-section benchmark. Wrap in a function so the work
    only runs when explicitly invoked.

    Sections:
      A: SINGLE-STUDY (Jiang 2015), 5-channel (PCA n=80)        — 121 samples
      B: SINGLE-STUDY 8-channel on 98-subset (PCA n=80)        — 98 samples
      C: CROSS-STUDY (pan-cancer), 5-channel (PCA n=200)       — 627 samples
      D: NAIVE 'HCC vs all healthy' (negative control)          — 627 samples
      E: CROSS-STUDY 8-channel (98-subset)                      — 98 samples
    """
    # === A: SINGLE-STUDY (Jiang 2015), 5-channel ===
    labels, studies = _load_labels(os.path.join(feat_dir, "labels.tsv"))
    # Older labels.tsv has no study column; force study='6' (Jiang)
    for s in list(studies):
        studies[s] = "6"
    X, y, st = load5(labels, studies, feat_dir)
    print(f"\n=== A: SINGLE-STUDY (Jiang 2015), 5-channel (PCA n=80) ===")
    print(f"  Cohort: {(y==1).sum()} cancer + {(y==0).sum()} healthy = {len(y)} total")
    run("5-channel (PCA n=80, harmonized)", X, y, st, 80, True)
    run("5-channel (PCA n=80, no harmonize)", X, y, st, 80, False)

    # === B: SINGLE-STUDY 8-channel on 98-subset ===
    X8, y8, st8 = load8(labels, studies, feat_dir)
    print(f"\n=== B: SINGLE-STUDY 8-channel on 98-subset (PCA n=80) ===")
    print(f"  Cohort: {(y8==1).sum()} cancer + {(y8==0).sum()} healthy = {len(y8)} total")
    run("8-channel (PCA n=80, harmonized)", X8, y8, st8, 80, True)
    run("8-channel (PCA n=200, harmonized)", X8, y8, st8, 200, True)

    # === C: CROSS-STUDY (pan-cancer), 5-channel, all 627 ===
    labels, studies = _load_labels(os.path.join(feat_dir, "labels_cross_study.tsv"))
    X, y, st = load5(labels, studies, feat_dir)
    print(f"\n=== C: CROSS-STUDY (pan-cancer), 5-channel (PCA n=200) ===")
    print(f"  Cohort: {(y==1).sum()} cancer + {(y==0).sum()} healthy = {len(y)} total, studies={set(st)}")
    run("5-channel (PCA n=200, harmonized)", X, y, st, 200, True)
    run("5-channel (PCA n=80, harmonized)", X, y, st, 80, True)
    run("5-channel (PCA n=200, no harmonize)", X, y, st, 200, False)

    # === D: NAIVE CROSS-STUDY (HCC vs all healthy negative control) ===
    labels_hcc, studies_hcc = {}, {}
    with open(os.path.join(feat_dir, "labels_cross_study.tsv")) as f:
        for line in f:
            p = line.strip().split("\t")
            s, l, st_ = p[0], p[1], p[2]
            if l == "cancer":
                labels_hcc[s] = 1; studies_hcc[s] = st_
            elif l == "healthy":
                labels_hcc[s] = 0; studies_hcc[s] = st_
    X, y, st = load5(labels_hcc, studies_hcc, feat_dir)
    print(f"\n=== D: NAIVE 'HCC vs all healthy' (negative control, PCA n=200) ===")
    print(f"  Cohort: {(y==1).sum()} cancer + {(y==0).sum()} healthy = {len(y)} total")
    run("5-channel (PCA n=200, harmonized)", X, y, st, 200, True)

    # === E: CROSS-STUDY 8-channel (98-subset) ===
    labels, studies = _load_labels(os.path.join(feat_dir, "labels_cross_study.tsv"))
    X8, y8, st8 = load8(labels, studies, feat_dir)
    if len(X8) >= 50:
        print(f"\n=== E: CROSS-STUDY 8-channel (98-subset) (PCA n=200) ===")
        print(f"  Cohort: {(y8==1).sum()} cancer + {(y8==0).sum()} healthy = {len(X8)} total")
        run("8-channel (PCA n=200, harmonized)", X8, y8, st8, 200, True)
    else:
        print(f"\n=== E: skipped (only {len(X8)} samples have all 8 channels) ===")


def main():
    """CLI entry point. Parses args then runs the full benchmark."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=FEAT,
                    help="Directory of {sample}.delfi_*.npy + .fsd.json "
                         "(default: data/features)")
    args = ap.parse_args()
    run_honest_benchmark(feat_dir=args.features_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
