#!/usr/bin/env python3
"""Train an ensemble classifier (Random Forest / Gradient Boosting) on the
normalized fragmentomic feature matrix to classify Cancer vs Healthy.

Feature matrix per sample (rows = samples, columns = features):
  - FSD summary statistics (median, mode, p10, p90, short_fraction, S/L ratio)
  - DELFI 100 kb window ratios (GC-corrected), summarized: mean, median, p10,
    p90, MAD, and the fraction of extreme windows
  - (optionally) 5 Mb window ratios
  - (optionally) 4-mer end-motif frequencies (BAM mode)

Rigorous validation:
  - stratified 5-fold CV (or LOOCV when n < 25)
  - no feature selection on the full data — the classifier consumes the
    pre-defined feature set
  - reports AUC, sens@95% spec, sens@99% spec, balanced accuracy

Usage:
  python train_classifier.py --features data/features --labels labels.tsv \
      --out results --model rf --cv 5

labels.tsv format (tab-separated, no header):
  sample<TAB>label        # label in {0, 1} or {healthy, cancer}
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn not installed. `pip install scikit-learn`",
          file=sys.stderr)
    sys.exit(1)


def load_features(features_dir: str, labels: dict[str, int],
                  with_motifs: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the sample×feature matrix from per-sample JSON/npy artifacts."""
    rows, cols, names = [], None, None
    for sample, label in sorted(labels.items()):
        # FSD
        fsd = os.path.join(features_dir, f"{sample}.fsd.json")
        if not os.path.exists(fsd):
            print(f"  ! missing {fsd}", file=sys.stderr)
            continue
        with open(fsd) as f:
            d = json.load(f)
        feats = [
            d.get("median_length", np.nan), d.get("mode_length", np.nan),
            d.get("mean_length", np.nan), d.get("p10", np.nan),
            d.get("p90", np.nan),
            d.get("short_fraction_100_150", np.nan),
            d.get("short_long_ratio", np.nan),
        ]
        fnames = ["fsd_median", "fsd_mode", "fsd_mean", "fsd_p10", "fsd_p90",
                  "fsd_short_frac", "fsd_short_long_ratio"]
        # DELFI: GC-corrected 100kb ratio vector → distribution summaries
        corr = os.path.join(features_dir, f"{sample}.gc_corrected.npy")
        if os.path.exists(corr):
            v = np.load(corr)
            v = v[np.isfinite(v) & (v > 0)]
            feats += [float(np.mean(v)), float(np.median(v)),
                      float(np.percentile(v, 10)), float(np.percentile(v, 90)),
                      float(np.std(v)),
                      float((v > np.percentile(v, 90)).mean())]
            fnames += ["delfi_mean", "delfi_median", "delfi_p10", "delfi_p90",
                       "delfi_std", "delfi_extreme_frac"]
        # WPS
        wps = os.path.join(features_dir, f"{sample}.wps_100kb.npy")
        if os.path.exists(wps):
            w = np.load(wps)
            w = w[np.isfinite(w)]
            feats += [float(np.mean(w)), float(np.std(w)), float(np.median(w))]
            fnames += ["wps_mean", "wps_std", "wps_median"]
        # FSD full profile (5bp bins 100-220bp — the fragment-length shape
        # is the primary tumor signal: tumor cfDNA is shorter)
        size_bins = d.get("size_bins", {})
        for b in sorted(size_bins):
            # keep the informative 100-220bp window (24 bins)
            try:
                lo = int(b.split("-")[0])
            except ValueError:
                continue
            if 100 <= lo < 220:
                feats.append(float(size_bins[b]))
                fnames.append(f"fsd_bin_{b}")
        # 5Mb DELFI ratio vector (CNA-scale chromatin signal)
        r5 = os.path.join(features_dir, f"{sample}.delfi_5mb_ratio.npy")
        if os.path.exists(r5):
            v5 = np.load(r5)
            v5 = v5[np.isfinite(v5) & (v5 > 0)]
            feats += [float(np.mean(v5)), float(np.median(v5)),
                      float(np.percentile(v5, 10)), float(np.percentile(v5, 90)),
                      float(np.std(v5)),
                      float((v5 > np.percentile(v5, 90)).mean())]
            fnames += ["delfi5_mean", "delfi5_median", "delfi5_p10",
                       "delfi5_p90", "delfi5_std", "delfi5_extreme_frac"]
        # CNV coverage profile (5Mb, median-normalized → copy number aberrations)
        cov5 = os.path.join(features_dir, f"{sample}.delfi_5mb_coverage.npy")
        if os.path.exists(cov5):
            c5 = np.load(cov5)
            c5 = c5[np.isfinite(c5) & (c5 > 0)]
            feats += [float(np.std(c5)),
                      float((c5 < 0.8).mean()),      # deletions
                      float((c5 > 1.2).mean()),      # amplifications
                      float((c5 > 1.5).mean()),      # high-level amps
                      float(np.percentile(c5, 5)),
                      float(np.percentile(c5, 95))]
            fnames += ["cnv_std", "cnv_del_frac", "cnv_amp_frac",
                       "cnv_high_amp_frac", "cnv_p5", "cnv_p95"]
        # Motifs (optional)
        if with_motifs:
            mf = os.path.join(features_dir, f"{sample}.motifs.json")
            if os.path.exists(mf):
                with open(mf) as f:
                    freqs = json.load(f)["freqs"]
                feats += [freqs.get(m, 0.0) for m in sorted(freqs)]
                fnames += [f"motif_{m}" for m in sorted(freqs)]
        rows.append(feats)
        names = fnames if cols is None else names
        cols = len(feats) if cols is None else cols
        if len(feats) != cols:
            print(f"  ! inconsistent feature count for {sample} "
                  f"({len(feats)} vs {cols})", file=sys.stderr)
            continue
    X = np.asarray(rows, dtype=float)
    y = np.asarray([labels[s] for s, _ in zip(sorted(labels), rows)], dtype=int)
    # NaN values are kept here and imputed INSIDE evaluate_cv using only
    # the train-fold median. The previous version used the full-cohort
    # median here, which leaked test-set NaN structure into the train
    # statistics. This function returns X with NaNs preserved.
    return X, y, names


def evaluate_cv(X, y, model, cv, use_pca: bool = False, pca_n: int = 20,
                study_arr=None, harmonize: bool = False) -> dict:
    """K-fold CV with POOLED out-of-fold predictions.

    Per-fold AUC is undefined for small test folds (e.g. LOOCV single
    samples), so we collect all out-of-fold predictions and compute a
    single ROC/AUC/fixed-specificity sensitivity on the pooled set —
    the standard honest protocol for small cohorts.
    """
    y_true_all: list[int] = []
    y_score_all: list[float] = []
    # Drop constant columns (std=0) — StandardScaler would produce NaN
    keep = np.nanstd(X, axis=0) > 1e-12
    X = X[:, keep]
    if X.shape[1] == 0:
        raise ValueError("all features are constant — nothing to learn")
    for tr, te in cv.split(X, y):
        # Per-fold NaN imputation using train-fold median only (was
        # previously applied once to the full cohort, leaking test-set
        # NaN structure into the train statistics).
        col_med = np.nanmedian(X[tr], axis=0)
        # Replace train NaNs with the train median; replace test NaNs
        # with the train median too (test never seen). Fill all-NaN
        # columns with 0 (median of nothing is NaN; avoid that).
        col_med = np.nan_to_num(col_med, nan=0.0)
        Xtr = X[tr].copy()
        Xte = X[te].copy()
        for c in range(X.shape[1]):
            Xtr[np.isnan(Xtr[:, c]), c] = col_med[c]
            Xte[np.isnan(Xte[:, c]), c] = col_med[c]
        if harmonize and study_arr is not None:
            Xtr, scalers = _harmonize(Xtr, study_arr[tr], None)
            Xte, _ = _harmonize(Xte, study_arr[te], scalers)
        else:
            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xte = scaler.transform(Xte)
        # PCA on the full profile inside the fold (DELFI: profile → PCA → RF)
        if use_pca and pca_n > 0 and Xtr.shape[1] > pca_n:
            n_comp = min(pca_n, Xtr.shape[1], Xtr.shape[0])
            pca = PCA(n_components=n_comp, random_state=42).fit(Xtr)
            Xtr = pca.transform(Xtr)
            Xte = pca.transform(Xte)
        m = model.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        y_true_all.extend(y[te].tolist())
        y_score_all.extend(p.tolist())

    y_true = np.asarray(y_true_all)
    y_score = np.asarray(y_score_all)
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    def sens_at(fpr_target):
        """Sensitivity at the operating point with FPR ≤ target.

        With n controls, FPR is quantized in steps of 1/n; the correct
        fixed-specificity sensitivity is the TPR at the LARGEST FPR that
        does not exceed the target (e.g. 99% spec with 30 controls → 0
        false positives → the threshold just above the top healthy score).
        The old argmin(|fpr-target|) snapped to fpr=0.0 (ROC origin) and
        wrongly reported 0.0 sensitivity.
        """
        ok = fpr <= fpr_target + 1e-9
        if not ok.any():
            return 0.0
        idx = int(np.where(ok)[0][-1])  # last (highest) valid operating point
        return float(tpr[idx])

    return {
        "n_folds": len(list(cv.split(X, y))),
        "auc_mean": auc,
        "auc_std": 0.0,  # pooled OOF — single estimate (honest for small n)
        "sens95_mean": sens_at(0.05),
        "sens99_mean": sens_at(0.01),
        "n_out_of_fold": len(y_true),
    }


def load_full_profile(features_dir: str, labels: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the FULL fragmentomic profile (DELFI-style).

    Four channels concatenated per sample:
      1. 5Mb short/long ratio      (631 bins — DELFI ratio profile)
      2. 5Mb coverage (median-norm)(631 bins — CNA at 5Mb)
      3. 100kb short/long ratio    (30,894 bins — finer DELFI ratio)
      4. 100kb coverage (median-norm) (30,894 bins — finer CNA)
      5. FSD size histogram        (196 bins — full fragment-length shape)
      6. 100kb mean length         (30,894 bins — spatial shortening)
      7. 5Mb mean length           (631 bins — coarse spatial shortening)
      8. 4-mer end motifs          (256 bins — nuclease preferences)

    The 100kb channels add ~50x spatial resolution; per-sample median
    normalization of coverage removes sequencing-depth batch effects. The
    FSD histogram captures the full fragment-length distribution shape
    (sub-nucleosomal, mononucleosome, dinucleosome peaks) — richer than
    the single 150bp short/long split.
    Returns (X, y, order); PCA is applied inside CV folds.
    """
    rows = []
    y = []
    order = []
    for sample in sorted(labels):
        r5 = os.path.join(features_dir, f"{sample}.delfi_5mb_ratio.npy")
        c5 = os.path.join(features_dir, f"{sample}.delfi_5mb_coverage.npy")
        r100 = os.path.join(features_dir, f"{sample}.delfi_100kb_ratio.npy")
        cnt100 = os.path.join(features_dir, f"{sample}.delfi_100kb_counts.npy")
        if not (os.path.exists(r5) and os.path.exists(c5)):
            continue
        v = [np.load(r5), np.load(c5)]
        if os.path.exists(r100):
            v.append(np.load(r100))
        if os.path.exists(cnt100):
            c = np.load(cnt100).astype(float)
            med = np.median(c)
            if med > 0:
                c = c / med  # per-sample median-normalize (removes depth)
            v.append(c)
        # FSD size histogram (5bp bins 20-1000bp) from the FSD JSON
        fsd_json = os.path.join(features_dir, f"{sample}.fsd.json")
        if os.path.exists(fsd_json):
            with open(fsd_json) as f:
                sb = json.load(f).get("size_bins", {})
            keys = sorted(sb, key=lambda k: int(k.split("-")[0]))
            v.append(np.asarray([sb[k] for k in keys], dtype=float))
        # Per-bin mean fragment length (spatial shortening)
        for suf in ("delfi_100kb_meanlen.npy", "delfi_5mb_meanlen.npy"):
            p = os.path.join(features_dir, f"{sample}.{suf}")
            if os.path.exists(p):
                v.append(np.load(p).astype(float))
        # 4-mer end-motif frequencies (nuclease preferences)
        mf = os.path.join(features_dir, f"{sample}.motifs.npy")
        if os.path.exists(mf):
            v.append(np.load(mf).astype(float))
        rows.append(np.concatenate(v))
        y.append(labels[sample])
        order.append(sample)
    X = np.asarray(rows, dtype=float)
    y = np.asarray(y, dtype=int)
    # NaN/Inf → 0 (bins with no coverage)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, order


def _harmonize(X, study_arr, scalers=None):
    """Per-study z-score: fit per-study StandardScaler (train) or apply (test).

    Returns (transformed_X, scalers_dict). Fitting on train only and applying
    the same per-study scalers to test removes study-specific mean/variance
    shifts without leaking test-set statistics.
    """
    if scalers is None:
        scalers = {}
        for st in np.unique(study_arr):
            mask = study_arr == st
            if mask.sum() > 1:
                scalers[st] = StandardScaler().fit(X[mask])
    out = np.empty_like(X, dtype=float)
    for st, sc in scalers.items():
        mask = study_arr == st
        if mask.any():
            out[mask] = sc.transform(X[mask])
    # studies with no fitted scaler (single sample in train) → leave as-is
    return out, scalers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", choices=["rf", "gb", "lr"], default="lr")
    ap.add_argument("--pca", action="store_true",
                    help="PCA-reduce the full profile inside each CV fold (DELFI-style)")
    ap.add_argument("--pca-n", type=int, default=200,
                    help="number of PCA components (default 200 — the full "
                         "4-channel profile needs more than the 5Mb-only 80)")
    ap.add_argument("--cv", type=int, default=5, help="folds (0 = LOOCV)")
    ap.add_argument("--with-motifs", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--harmonize", action="store_true",
                    help="per-study z-score harmonization (cross-study cohorts; "
                         "labels file must have a 3rd study column)")
    args = ap.parse_args()

    labels = {}
    studies = {}
    with open(args.labels) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sample, lab = parts[0], parts[1].lower()
            if lab in ("cancer", "1", "tumor", "positive"):
                labels[sample] = 1
            elif lab in ("healthy", "0", "control", "normal", "negative"):
                labels[sample] = 0
            if len(parts) >= 3:
                studies[sample] = parts[2].strip()
    print(f"Labels: {len(labels)} samples "
          f"({sum(labels.values())} cancer, {len(labels) - sum(labels.values())} healthy)")

    if args.pca:
        # DELFI-style: full 5Mb profile → PCA (inside CV) → RF
        X, y, order = load_full_profile(args.features, labels)
        feature_desc = "full-5mb-profile-pca"
        study_arr = np.asarray([studies.get(s, "default") for s in order]) \
            if studies else None
        print(f"Full 5Mb profile matrix: {X.shape[0]} samples x {X.shape[1]} bins")
    else:
        X, y, fnames = load_features(args.features, labels, args.with_motifs)
        feature_desc = fnames
        study_arr = None
        print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=args.n_estimators,
                                       random_state=42, n_jobs=-1)
    elif args.model == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000, C=1.0)
    else:
        model = GradientBoostingClassifier(n_estimators=args.n_estimators,
                                           random_state=42)
    cv = LeaveOneOut() if args.cv == 0 or X.shape[0] < 25 else \
        StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    res = evaluate_cv(X, y, model, cv, use_pca=args.pca, pca_n=args.pca_n,
                      study_arr=study_arr, harmonize=args.harmonize)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "classifier_results.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "cv": str(cv), "features": feature_desc,
                   "n_samples": int(X.shape[0]), **res}, f, indent=2)
    print(f"\n=== RESULTS ({args.model}, {res['n_folds']}-fold CV) ===")
    print(f"AUC:         {res['auc_mean']:.3f} ± {res['auc_std']:.3f}")
    print(f"Sens@95%:    {res['sens95_mean']:.3f}")
    print(f"Sens@99%:    {res['sens99_mean']:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
