"""
Honest test: do non-linear models beat LR-on-PCA(200) on the 627 cohort?

note: I (the model) am running this experiment because the user
asked to keep iterating on code experiments. I previously flagged
that further feature engineering is at diminishing returns.
This experiment tests whether non-linear methods can pick up signal
that the LR-on-PCA linear classifier misses.

Pre-registered prediction: non-linear methods will likely UNDERPERFORM
LR-on-PCA(200) on this dataset size (627 samples, 63k features). Reasons:
- RF/GB tend to overfit on high-dim/low-sample data without aggressive
  regularization
- LR-on-PCA(200) is itself a strong, regularized baseline
- LR has fewer parameters to overfit
- Non-linear methods may find spurious interactions that don't generalize

If non-linear methods UNDERPERFORM by a large margin: confirms
LR-on-PCA is the right baseline and the linear signal is the
dominant information in the data.

If non-linear methods BEAT LR-on-PCA: a small but real signal exists
in non-linear interactions, suggesting deep learning or kernel SVM
might gain more.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from train_classifier import _harmonize  # noqa

FEAT_DIR = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features"
LABELS_TSV = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features/labels_cross_study.tsv"


def _load():
    samples, y, studies = [], {}, {}
    with open(LABELS_TSV) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            samples.append(parts[0])
            y[parts[0]] = 1 if parts[1].lower() in ("cancer", "1", "tumor") else 0
            studies[parts[0]] = parts[2] if len(parts) >= 3 else "unknown"
    return samples, np.asarray([y[s] for s in samples], dtype=int), studies


def _build_X(samples):
    rows = []
    for s in samples:
        r5 = np.load(os.path.join(FEAT_DIR, f"{s}.delfi_5mb_ratio.npy"))
        c5 = np.load(os.path.join(FEAT_DIR, f"{s}.delfi_5mb_coverage.npy"))
        r100 = np.load(os.path.join(FEAT_DIR, f"{s}.delfi_100kb_ratio.npy"))
        c100_raw = np.load(os.path.join(FEAT_DIR,
                                          f"{s}.delfi_100kb_counts.npy"))
        c100 = c100_raw / np.median(c100_raw)
        with open(os.path.join(FEAT_DIR, f"{s}.fsd.json")) as f:
            d = json.load(f)
        keys = sorted(d["size_bins"].keys(), key=lambda k: int(k.split("-")[0]))
        fsd = np.asarray([d["size_bins"][k] for k in keys], dtype=float)
        rows.append(np.concatenate([r5, c5, r100, c100, fsd]))
    X = np.nan_to_num(np.stack(rows).astype(float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    return X


def _evaluate(X, y, study, model_factory, name, n_seeds=3):
    """5-fold CV with harmonization; returns per-seed AUC list."""
    aucs = []
    for s in range(n_seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            Xtr_h, scalers = _harmonize(X[tr], study[tr], None)
            Xte_h, _ = _harmonize(X[te], study[te], scalers)
            m = model_factory()
            m.fit(Xtr_h, y[tr])
            if hasattr(m, "predict_proba"):
                ys.extend(m.predict_proba(Xte_h)[:, 1].tolist())
            else:
                ys.extend(m.decision_function(Xte_h).tolist())
            yt.extend(y[te].tolist())
        try:
            aucs.append(roc_auc_score(yt, ys))
        except Exception:
            pass
    return {"name": name, "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)), "per_seed_aucs": aucs}


def main():
    print("[model_ablation] Loading cohort...")
    samples, y, studies = _load()
    ok = []
    for s in samples:
        if all(os.path.exists(os.path.join(FEAT_DIR, f"{s}.{x}.npy"))
                for x in ("delfi_5mb_ratio", "delfi_5mb_coverage",
                           "delfi_100kb_ratio", "delfi_100kb_counts")) \
                and os.path.exists(os.path.join(FEAT_DIR, f"{s}.fsd.json")):
            ok.append(s)
    samples = ok
    y = np.asarray([studies[s] if False else (1 if studies.get(s) == "cristiano" else 2)  # placeholder
                     for s in samples])  # not used; we keep study strings
    study = np.asarray([studies[s] for s in samples])
    # Re-load y properly
    y_label = {}
    with open(LABELS_TSV) as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if parts[0] in samples:
                y_label[parts[0]] = 1 if parts[1].lower() == "cancer" else 0
    y = np.asarray([y_label[s] for s in samples], dtype=int)
    print(f"[model_ablation] {len(samples)} samples, {y.sum()} cancer, "
          f"{(y == 0).sum()} healthy")

    print("[model_ablation] Building feature matrix...")
    X = _build_X(samples)
    print(f"[model_ablation] X shape: {X.shape}")

    # Drop constant columns (mirrors honest_benchmark.py)
    keep = np.nanstd(X, axis=0) > 1e-12
    X = X[:, keep]
    print(f"[model_ablation] After dropping constant columns: {X.shape}")

    results = []

    # Baseline: LR (no PCA — direct on harmonized features)
    print("\n[model_ablation] LR (no PCA, no reg)...")
    t0 = time.time()
    r = _evaluate(X, y, study, lambda: LogisticRegression(max_iter=2000, C=1.0),
                   "LR (no PCA)")
    print(f"  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)

    # Baseline: LR + PCA(200)
    print("\n[model_ablation] LR + PCA(200)...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for s in range(5):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            Xtr_h = _harmonize(X[tr], study[tr], None)[0]
            Xte_h = _harmonize(X[te], study[te], _harmonize(X[tr], study[tr], None)[1])[0]
            sc = StandardScaler().fit(Xtr_h)
            Xtr_s = sc.transform(Xtr_h); Xte_s = sc.transform(Xte_h)
            n_comp = min(200, Xtr_s.shape[0], Xtr_s.shape[1])
            pca = PCA(n_components=n_comp, random_state=0).fit(Xtr_s)
            m = LogisticRegression(max_iter=20000, tol=1e-8, random_state=0)
            m.fit(pca.transform(Xtr_s), y[tr])
            ys.extend(m.predict_proba(pca.transform(Xte_s))[:, 1].tolist())
            yt.extend(y[te].tolist())
        aucs.append(roc_auc_score(yt, ys))
    r = {"name": "LR + PCA(200)", "auc_mean": float(np.mean(aucs)),
         "auc_std": float(np.std(aucs)), "per_seed_aucs": aucs}
    print(f"  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    results.append(r)

    # Random Forest
    print("\n[model_ablation] RandomForest(n_estimators=200, max_depth=6)...")
    t0 = time.time()
    r = _evaluate(X, y, study,
                   lambda: RandomForestClassifier(
                       n_estimators=200, max_depth=6,
                       min_samples_leaf=5, n_jobs=-1, random_state=42),
                   "RF(200, d=6)")
    print(f"  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)

    # Gradient Boosting
    print("\n[model_ablation] GradientBoosting(n=100, depth=3)...")
    t0 = time.time()
    r = _evaluate(X, y, study,
                   lambda: GradientBoostingClassifier(
                       n_estimators=100, max_depth=3,
                       learning_rate=0.05, random_state=42),
                   "GB(100, d=3)")
    print(f"  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)

    out = {"results": results, "n_samples": len(samples)}
    with open("/tmp/model_ablation.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'model':<30} {'AUC':>10} {'±':>5} {'std':>5}")
    for r in results:
        print(f"{r['name']:<30} {r['auc_mean']:>10.4f} {r['auc_std']:>5.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
