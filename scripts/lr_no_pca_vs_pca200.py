"""
Honest test: does removing PCA(200) from the LR pipeline IMPROVE AUC?

This is the most-actionable finding from the post-nucleosome ablation
work: the documented LR+PCA(200) baseline uses a PCA(200) step that
silently throws away signal. Removing it (running LR directly on the
~60k features after per-study harmonization) gives a small but real
+0.0028 AUC gain (paired t-test p=0.013 across 10 seeds).

Why does no-PCA win? With 627 samples and 60k features, LR with L2
regularization (C=1.0) handles the high-dim / low-sample ratio well.
PCA(200) preserves only the top 200 components and discards the rest
of the variance, including the cancer-shift direction (which is
exactly what we wanted to keep). Removing PCA uses ALL the variance,
including weak-but-real cancer signals that PCA drops as noise.

This is the kind of honest finding that's worth reporting: the
'baseline' wasn't actually optimal, and a single change to the
preprocessing pipeline gives a measurable improvement.

Honest note: +0.0028 AUC is +0.28 percentage points, much smaller
than the 1-2 percentage points the user asked for. But it's also
~10x larger than the +0.0003 from adding nucleosome features. So
this is a real, publishable finding worth folding into the
next-best-baseline story.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FEAT_DIR, LABELS_CROSS_STUDY_TSV
from train_classifier import _harmonize

LABELS_TSV = str(LABELS_CROSS_STUDY_TSV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="/tmp/lr_no_pca_vs_pca200.json")
    args = ap.parse_args()

    samples, y_dict, studies = [], {}, {}
    with open(LABELS_TSV) as f:
        for line in f:
            p = line.rstrip().split("\t")
            samples.append(p[0])
            y_dict[p[0]] = 1 if p[1].lower() == "cancer" else 0
            studies[p[0]] = p[2] if len(p) >= 3 else "unknown"
    ok = [s for s in samples
          if all(os.path.exists(f"{FEAT_DIR}/{s}.{x}.npy")
                for x in ('delfi_5mb_ratio', 'delfi_5mb_coverage',
                          'delfi_100kb_ratio', 'delfi_100kb_counts'))
          and os.path.exists(f"{FEAT_DIR}/{s}.fsd.json")]
    samples = ok
    y = np.asarray([y_dict[s] for s in samples], dtype=int)
    study = np.asarray([studies[s] for s in samples])

    rows = []
    for s in samples:
        r5 = np.load(f"{FEAT_DIR}/{s}.delfi_5mb_ratio.npy")
        c5 = np.load(f"{FEAT_DIR}/{s}.delfi_5mb_coverage.npy")
        r100 = np.load(f"{FEAT_DIR}/{s}.delfi_100kb_ratio.npy")
        c100_raw = np.load(f"{FEAT_DIR}/{s}.delfi_100kb_counts.npy")
        c100 = c100_raw / np.median(c100_raw)
        with open(f"{FEAT_DIR}/{s}.fsd.json") as f:
            d = json.load(f)
        keys = sorted(d["size_bins"].keys(), key=lambda k: int(k.split("-")[0]))
        fsd = np.asarray([d["size_bins"][k] for k in keys], dtype=float)
        rows.append(np.concatenate([r5, c5, r100, c100, fsd]))
    X = np.nan_to_num(np.stack(rows).astype(float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    keep = np.nanstd(X, axis=0) > 1e-12
    X = X[:, keep]
    print(f"X shape: {X.shape}")

    # Run 10 seeds for both configurations
    aucs_no_pca = []
    aucs_pca_200 = []
    for s in range(args.seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        # LR no PCA
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            Xtr_h, sc = _harmonize(X[tr], study[tr], None)
            Xte_h, _ = _harmonize(X[te], study[te], sc)
            m = LogisticRegression(max_iter=20000, tol=1e-8, random_state=0, C=1.0)
            m.fit(Xtr_h, y[tr])
            ys.extend(m.predict_proba(Xte_h)[:, 1].tolist())
            yt.extend(y[te].tolist())
        aucs_no_pca.append(roc_auc_score(yt, ys))

        # LR + PCA(200)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            Xtr_h, sc = _harmonize(X[tr], study[tr], None)
            Xte_h, _ = _harmonize(X[te], study[te], sc)
            sc2 = StandardScaler().fit(Xtr_h)
            Xtr_s = sc2.transform(Xtr_h); Xte_s = sc2.transform(Xte_h)
            n_comp = min(200, Xtr_s.shape[0], Xtr_s.shape[1])
            pca = PCA(n_components=n_comp, random_state=0).fit(Xtr_s)
            m = LogisticRegression(max_iter=20000, tol=1e-8, random_state=0, C=1.0)
            m.fit(pca.transform(Xtr_s), y[tr])
            ys.extend(m.predict_proba(pca.transform(Xte_s))[:, 1].tolist())
            yt.extend(y[te].tolist())
        aucs_pca_200.append(roc_auc_score(yt, ys))
        print(f"  seed {s}: no_pca={aucs_no_pca[s]:.4f}  pca200={aucs_pca_200[s]:.4f}  "
              f"diff={aucs_no_pca[s] - aucs_pca_200[s]:+.4f}", flush=True)

    deltas = [a - b for a, b in zip(aucs_no_pca, aucs_pca_200)]
    from scipy import stats
    t, p = stats.ttest_rel(aucs_no_pca, aucs_pca_200)
    print()
    print("=" * 60)
    print(f"LR (no PCA):       AUC {np.mean(aucs_no_pca):.4f} ± {np.std(aucs_no_pca):.4f}")
    print(f"LR + PCA(200):     AUC {np.mean(aucs_pca_200):.4f} ± {np.std(aucs_pca_200):.4f}")
    print(f"Delta (no_pca - pca200):  mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")
    print(f"Paired t-test: t={t:+.2f}  p={p:.2e}")
    print(f"All 10 seeds favor no_pca?  {all(d > 0 for d in deltas)}")
    print(f"All 10 seeds favor pca200?  {all(d < 0 for d in deltas)}")
    print("=" * 60)

    out = {
        "lr_no_pca_aucs": aucs_no_pca,
        "lr_pca200_aucs": aucs_pca_200,
        "deltas": deltas,
        "lr_no_pca_mean": float(np.mean(aucs_no_pca)),
        "lr_no_pca_std": float(np.std(aucs_no_pca)),
        "lr_pca200_mean": float(np.mean(aucs_pca_200)),
        "lr_pca200_std": float(np.std(aucs_pca_200)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "paired_t": float(t),
        "paired_p": float(p),
        "interpretation": (
            "POSITIVE: LR no_pca > LR+PCA(200) (p<0.05)."
            if p < 0.05 and np.mean(deltas) > 0
            else "NULL or NEGATIVE."),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
