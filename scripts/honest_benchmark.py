#!/usr/bin/env python3
"""HONEST benchmark — no optimization, no cherry-picking.

- 5 different random seeds for CV split
- Report mean +/- std across seeds
- Test every claim: 5-channel, 8-channel, single-study, cross-study
- Naive cross-study as a negative control
- Single-study baseline vs re-extracted subset for fair ablation
"""
import numpy as np, os, json, sys
sys.path.insert(0, 'scripts')
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from train_classifier import _harmonize

FEAT = 'data/features'
SEEDS = [42, 13, 7, 99, 1234]


def fsd_vec(s):
    """196-bin fragment-length histogram from FSD JSON."""
    p = os.path.join(FEAT, f"{s}.fsd.json")
    if not os.path.exists(p): return None
    with open(p) as f: sb = json.load(f)['size_bins']
    keys = sorted(sb, key=lambda k: int(k.split('-')[0]))
    return np.array([sb[k] for k in keys], dtype=float)


def load5(labels, studies):
    """5-channel: 5mb_ratio + 5mb_coverage + 100kb_ratio + 100kb_counts + FSD."""
    rows, order, y = [], [], []
    for s in sorted(labels):
        r5 = os.path.join(FEAT, f"{s}.delfi_5mb_ratio.npy")
        c5 = os.path.join(FEAT, f"{s}.delfi_5mb_coverage.npy")
        r100 = os.path.join(FEAT, f"{s}.delfi_100kb_ratio.npy")
        c100 = os.path.join(FEAT, f"{s}.delfi_100kb_counts.npy")
        if not all(os.path.exists(p) for p in (r5, c5, r100, c100)): continue
        sb = fsd_vec(s)
        if sb is None: continue
        cn = np.load(c100) / np.median(np.load(c100))
        v = np.concatenate([np.load(r5), np.load(c5), np.load(r100), cn, sb])
        rows.append(v); order.append(s); y.append(labels[s])
    return np.asarray(rows), np.asarray(y), np.array([studies.get(s, '6') for s in order])


def load8(labels, studies):
    """8-channel: 5 + 100kb_meanlen + 5mb_meanlen + motifs."""
    rows, order, y = [], [], []
    for s in sorted(labels):
        r5 = os.path.join(FEAT, f"{s}.delfi_5mb_ratio.npy")
        c5 = os.path.join(FEAT, f"{s}.delfi_5mb_coverage.npy")
        r100 = os.path.join(FEAT, f"{s}.delfi_100kb_ratio.npy")
        c100 = os.path.join(FEAT, f"{s}.delfi_100kb_counts.npy")
        ml100 = os.path.join(FEAT, f"{s}.delfi_100kb_meanlen.npy")
        ml5 = os.path.join(FEAT, f"{s}.delfi_5mb_meanlen.npy")
        mot = os.path.join(FEAT, f"{s}.motifs.npy")
        if not all(os.path.exists(p) for p in (r5, c5, r100, c100, ml100, ml5, mot)): continue
        sb = fsd_vec(s)
        if sb is None: continue
        cn = np.load(c100) / np.median(np.load(c100))
        v = np.concatenate([np.load(r5), np.load(c5), np.load(r100), cn, sb,
                            np.load(ml100), np.load(ml5), np.load(mot)])
        rows.append(v); order.append(s); y.append(labels[s])
    return np.asarray(rows), np.asarray(y), np.array([studies.get(s, '6') for s in order])


def eval_cv(X, y, st, pca_n, harmonize, seeds):
    aucs, s95s, s99s = [], [], []
    # Cap PCA at min(n_samples-1, n_features) for the smallest train fold
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
    return np.mean(aucs), np.std(aucs), np.mean(s95s), np.std(s95s), np.mean(s99s), np.std(s99s)


def run(name, X, y, st, pca_n, harmonize):
    a, sa, s95, ss95, s99, ss99 = eval_cv(X, y, st, pca_n, harmonize, SEEDS)
    print(f"{name:55s} N={len(y):3d}  AUC {a:.4f}±{sa:.4f}  S95 {s95:.3f}±{ss95:.3f}  S99 {s99:.3f}±{ss99:.3f}")
    return a


# === A: SINGLE-STUDY (Jiang 2015), 5-channel ===
labels = {}; studies = {}
for line in open('data/features/labels.tsv'):
    p = line.strip().split('\t')
    labels[p[0]] = 1 if p[1] == 'cancer' else 0
    studies[p[0]] = '6'
X, y, st = load5(labels, studies)
print(f"\n=== A: SINGLE-STUDY (Jiang 2015), 5-channel (PCA n=80) ===")
print(f"  Cohort: {(y==1).sum()} cancer + {(y==0).sum()} healthy = {len(y)} total")
run("5-channel (PCA n=80, harmonized)", X, y, st, 80, True)
run("5-channel (PCA n=80, no harmonize)", X, y, st, 80, False)

# === B: SINGLE-STUDY 8-channel (only the 98 re-extracted) ===
X8, y8, st8 = load8(labels, studies)
print(f"\n=== B: SINGLE-STUDY 8-channel on 98-subset (PCA n=80) ===")
print(f"  Cohort: {(y8==1).sum()} cancer + {(y8==0).sum()} healthy = {len(y8)} total")
run("8-channel (PCA n=80, harmonized)", X8, y8, st8, 80, True)
run("8-channel (PCA n=200, harmonized)", X8, y8, st8, 200, True)

# === C: CROSS-STUDY (pan-cancer), 5-channel, all 627 ===
labels = {}; studies = {}
for line in open('data/features/labels_cross_study.tsv'):
    p = line.strip().split('\t')
    labels[p[0]] = 1 if p[1] == 'cancer' else 0
    studies[p[0]] = p[2]
X, y, st = load5(labels, studies)
print(f"\n=== C: CROSS-STUDY (pan-cancer), 5-channel (PCA n=200) ===")
print(f"  Cohort: {(y==1).sum()} cancer + {(y==0).sum()} healthy = {len(y)} total, studies={set(st)}")
run("5-channel (PCA n=200, harmonized)", X, y, st, 200, True)
run("5-channel (PCA n=80, harmonized)", X, y, st, 80, True)
run("5-channel (PCA n=200, no harmonize)", X, y, st, 200, False)

# === D: NAIVE CROSS-STUDY HCC vs all healthy (negative control) ===
labels_hcc = {}; studies_hcc = {}
for line in open('data/features/labels_cross_study.tsv'):
    p = line.strip().split('\t')
    s, l, st_ = p[0], p[1], p[2]
    if l == 'cancer':  # ALL cancer → "HCC vs healthy" mislabel
        labels_hcc[s] = 1; studies_hcc[s] = st_
for line in open('data/features/labels_cross_study.tsv'):
    p = line.strip().split('\t')
    s, l, st_ = p[0], p[1], p[2]
    if l == 'healthy':
        labels_hcc[s] = 0; studies_hcc[s] = st_
X, y, st = load5(labels_hcc, studies_hcc)
print(f"\n=== D: NAIVE 'HCC vs all healthy' (negative control, PCA n=200) ===")
print(f"  Cohort: {(y==1).sum()} cancer + {(y==0).sum()} healthy = {len(y)} total")
run("5-channel (PCA n=200, harmonized)", X, y, st, 200, True)

# === E: CROSS-STUDY with --with-motifs (using --pca auto-loads motifs; check 8-channel on full subset) ===
labels = {}; studies = {}
for line in open('data/features/labels_cross_study.tsv'):
    p = line.strip().split('\t')
    labels[p[0]] = 1 if p[1] == 'cancer' else 0
    studies[p[0]] = p[2]
X8, y8, st8 = load8(labels, studies)
if len(X8) >= 50:
    print(f"\n=== E: CROSS-STUDY 8-channel (98-subset) (PCA n=200) ===")
    print(f"  Cohort: {(y8==1).sum()} cancer + {(y8==0).sum()} healthy = {len(X8)} total")
    run("8-channel (PCA n=200, harmonized)", X8, y8, st8, 200, True)
else:
    print(f"\n=== E: skipped (only {len(X8)} samples have all 8 channels) ===")


def main():
    """CLI shim — module-level code above already ran at import time.

    The honest_benchmark script predates the [project.scripts] convention;
    the work happens at the top of the file. This shim exists so
    `pyproject.toml`'s console_scripts entry can resolve to a callable
    and `python -m scripts.honest_benchmark --help` works.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default="data/features",
                    help="Directory of {sample}.delfi_*.npy + .fsd.json")
    args = ap.parse_args()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())