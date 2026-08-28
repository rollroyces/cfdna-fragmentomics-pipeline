"""8-channel evaluation on the full cross-study cohort.

This script answers the question: "would adding 4-mer motif features
and per-bin mean-length features improve AUC on the full 627-sample
cross-study cohort?"

The honest answer is: only 98 samples have motif + mean-length files
on disk. The remaining 529 samples need motif + mean-length extraction
from their fragment TSV files (~1-2 hour extraction job).

For now, this script runs both:
  1. The 8-channel AUC on the 98-sample subset (where features exist)
  2. The 5-channel AUC on the same 98-sample subset (apples-to-apples)
  3. The 5-channel AUC on the full 627-sample cohort (full-cohort baseline)

So we can answer: "On the 98-sample subset, does 8-channel beat
5-channel?" which is the closest we can get to the S4 question
without re-extracting features.

Historical context (from BENCHMARK.md Appendix B line 91-95):
  "4-mer motifs add +0.005 AUC (98-sample subset only, n=50 cancer + 48 healthy)"

This script reproduces and extends that finding with proper C=1000
regularization (matches the headline recommended config).

Output: results/8channel_98subset.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FEAT_DIR
from train_classifier import _harmonize  # noqa

FEAT = str(FEAT_DIR)
SEEDS = [42, 13, 7, 99, 1234]


def fsd_vec(s):
    p = os.path.join(FEAT, f"{s}.fsd.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        sb = json.load(f)["size_bins"]
    keys = sorted(sb, key=lambda k: int(k.split("-")[0]))
    return np.array([sb[k] for k in keys], dtype=float)


def load_channels(samples, studies, channels=("5mb_ratio", "5mb_coverage",
                                                "100kb_ratio", "100kb_counts",
                                                "fsd")):
    """Load the requested channels; skip samples missing any.

    Returns: (X, y, st, sample_ids_kept) where sample_ids_kept is the
    ordered list of samples that survived the missing-feature filter.
    Use this to align downstream subsets.
    """
    rows, order, y, st = [], [], [], []
    needed_paths = []
    for s in sorted(samples):
        paths = []
        for ch in channels:
            if ch == "fsd":
                if not os.path.exists(os.path.join(FEAT, f"{s}.fsd.json")):
                    paths = None; break
            else:
                p = os.path.join(FEAT, f"{s}.delfi_{ch}.npy")
                if not os.path.exists(p):
                    paths = None; break
                paths.append(p)
        if paths is None:
            continue
        if "fsd" in channels:
            sb = fsd_vec(s)
            if sb is None:
                continue
        vecs = []
        for ch, p in zip(channels, paths):
            if ch == "100kb_counts":
                cn = np.load(p) / np.median(np.load(p))
                vecs.append(cn)
            else:
                vecs.append(np.load(p))
        if "fsd" in channels:
            vecs.append(sb)
        v = np.concatenate(vecs)
        rows.append(v)
        order.append(s)
        y.append(samples[s])
        st.append(studies.get(s, "unknown"))
    return (np.asarray(rows), np.asarray(y), np.asarray(st), order)


def load_extra(samples, studies, extras):
    """Load additional channels (mean-length bins + motif 256-bin)."""
    rows, order, y = [], [], []
    for s in sorted(samples):
        paths = []
        for ch in extras:
            p = os.path.join(FEAT, f"{s}.delfi_{ch}.npy") if ch != "motifs" \
                else os.path.join(FEAT, f"{s}.motifs.npy")
            if not os.path.exists(p):
                paths = None; break
            paths.append(p)
        if paths is None:
            continue
        v = np.concatenate([np.load(p) for p in paths])
        rows.append(v); order.append(s); y.append(samples[s])
    return (np.asarray(rows), np.asarray(y),
            np.array([studies.get(s, "unknown") for s in order]))


def evaluate(X, y, st, C, seeds, name):
    """Run LR no-PCA at C; report mean ± std AUC over seeds."""
    aucs = []
    for seed in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            Xtr_h, sc = _harmonize(X[tr], st[tr], None)
            Xte_h, _ = _harmonize(X[te], st[te], sc)
            m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                   max_iter=20000, tol=1e-8, random_state=0)
            m.fit(Xtr_h, y[tr])
            ys.extend(m.predict_proba(Xte_h)[:, 1].tolist())
            yt.extend(y[te].tolist())
        aucs.append(roc_auc_score(yt, ys))
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    print(f"  {name:55s} N={len(y):3d}  AUC {auc_mean:.4f} ± {auc_std:.4f}")
    return {"auc_mean": auc_mean, "auc_std": auc_std, "per_seed_aucs": aucs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/8channel_98subset.json")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    seeds = list(range(args.seeds))

    # Load cross-study labels
    labels, studies = {}, {}
    with open(f"{FEAT}/labels_cross_study.tsv") as f:
        for line in f:
            p = line.strip().split("\t")
            labels[p[0]] = 1 if p[1] == "cancer" else 0
            studies[p[0]] = p[2]

    # 5-channel: 5 + 100kb_counts + FSD
    print("=== 5-channel baseline ===")
    X5, y5, st5, kept5 = load_channels(labels, studies,
                                       channels=("5mb_ratio", "5mb_coverage",
                                                 "100kb_ratio", "100kb_counts", "fsd"))
    print(f"  Loaded 5-channel: {X5.shape[0]} samples × {X5.shape[1]} features")

    # 8-channel: 5 + 100kb_meanlen + 5mb_meanlen + motifs (only 98 samples)
    extras = ("100kb_meanlen", "5mb_meanlen", "motifs")
    X_extra, y_extra, st_extra = load_extra(labels, studies, extras)
    print(f"  Loaded extras ({', '.join(extras)}): {X_extra.shape[0]} samples × {X_extra.shape[1]} features")

    # Restrict to samples that have both 5-channel and extras (intersection)
    samples_with_extras = set()
    with open(f"{FEAT}/labels_cross_study.tsv") as f:
        for line in f:
            p = line.strip().split("\t")
            s = p[0]
            if all(os.path.exists(f"{FEAT}/{s}.delfi_{ch}.npy") for ch in extras[:-1]) \
                    and os.path.exists(f"{FEAT}/{s}.motifs.npy"):
                samples_with_extras.add(s)

    # Restrict 5-channel to the 98-sample subset (use the actual row
    # order from load_channels, not from labels dict)
    kept5_set = set(kept5)
    X5_sub_idx = [i for i, s in enumerate(kept5) if s in samples_with_extras]
    X5_sub = X5[X5_sub_idx]
    y5_sub = y5[X5_sub_idx]
    st5_sub = st5[X5_sub_idx]
    print(f"  5-channel restricted to 98-subset: {X5_sub.shape[0]} samples")

    # Concatenate 5 + extras for 8-channel. Use kept5 order so the
    # 8-channel rows are aligned with the 5-channel subset rows.
    samples_in_order = [s for s in kept5 if s in samples_with_extras]
    X8_list, y8_list, st8_list = [], [], []
    for s in samples_in_order:
        r5 = np.load(f"{FEAT}/{s}.delfi_5mb_ratio.npy")
        c5 = np.load(f"{FEAT}/{s}.delfi_5mb_coverage.npy")
        r100 = np.load(f"{FEAT}/{s}.delfi_100kb_ratio.npy")
        c100 = np.load(f"{FEAT}/{s}.delfi_100kb_counts.npy")
        cn = c100 / np.median(c100)
        sb = fsd_vec(s)
        ml100 = np.load(f"{FEAT}/{s}.delfi_100kb_meanlen.npy")
        ml5 = np.load(f"{FEAT}/{s}.delfi_5mb_meanlen.npy")
        mot = np.load(f"{FEAT}/{s}.motifs.npy")
        v = np.concatenate([r5, c5, r100, cn, sb, ml100, ml5, mot])
        X8_list.append(v)
        y8_list.append(1 if labels[s] == 1 else 0)
        st8_list.append(studies[s])
    X8 = np.asarray(X8_list)
    y8 = np.asarray(y8_list)
    st8 = np.asarray(st8_list)
    print(f"  8-channel concatenated: {X8.shape[0]} samples × {X8.shape[1]} features")
    assert X5_sub.shape[0] == X8.shape[0], (
        f"subset mismatch: 5ch={X5_sub.shape[0]} 8ch={X8.shape[0]}")
    # Verify alignment
    for i in range(min(5, X5_sub.shape[0])):
        assert kept5[X5_sub_idx[i]] == samples_in_order[i], (
            f"alignment mismatch at row {i}: 5ch={kept5[X5_sub_idx[i]]}, "
            f"8ch={samples_in_order[i]}")
    print(f"  Sanity: 5ch-shape={X5_sub.shape}, 8ch-shape={X8.shape}, "
          f"alignment verified")

    # Run evaluations
    results = {}
    print("\n=== Evaluation (LR no-PCA at C=1000, 5 seeds × 5-fold CV) ===\n")
    print("--- On 98-sample subset ---")
    results["5ch_98sub"] = evaluate(X5_sub, y5_sub, st5_sub, C=1000, seeds=seeds,
                                     name="5-channel (98-subset, n=98)")
    results["8ch_98sub"] = evaluate(X8, y8, st8, C=1000, seeds=seeds,
                                     name="8-channel (98-subset, n=98)")

    print("\n--- Full 627 cohort (5-channel only; 8-channel features not on disk for 529 samples) ---")
    results["5ch_full"] = evaluate(X5, y5, st5, C=1000, seeds=seeds,
                                    name="5-channel (full cohort, n=627)")

    # Paired t-test for 8ch vs 5ch on the 98-subset
    from scipy import stats
    if len(results["8ch_98sub"]["per_seed_aucs"]) >= 2:
        diff = np.array(results["8ch_98sub"]["per_seed_aucs"]) - np.array(results["5ch_98sub"]["per_seed_aucs"])
        t, p = stats.ttest_rel(results["8ch_98sub"]["per_seed_aucs"],
                                results["5ch_98sub"]["per_seed_aucs"])
        print(f"\n8ch - 5ch paired difference on 98-subset: {diff.mean():+.4f} ± {diff.std():.4f}")
        print(f"  Paired t = {t:+.3f}, p = {p:.4f}")

        results["8ch_vs_5ch_98sub_paired_t"] = {
            "delta_mean": float(diff.mean()),
            "delta_std": float(diff.std()),
            "t_stat": float(t),
            "p_value": float(p),
        }

    out = {
        "n_8channel_samples": int(X8.shape[0]),
        "n_5channel_full_samples": int(X5.shape[0]),
        "results": results,
        "note": (
            "8-channel features (4-mer motifs + per-bin mean length) "
            "are available for only 98 of 627 cross-study samples. "
            "Re-extracting these features for the other 529 samples "
            "would require running extract_motifs.py + a mean-length "
            "extractor on each .frag.tsv (~1-2 hour job)."
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
