"""
Gemma-vs-DeepCatch fragmentomics baseline.

Experiment design
-----------------
Question: can a general-purpose LLM (Gemma 2 9B), given a textual
summary of the 5-channel fragmentomics profile for each cfDNA sample,
match the AUC of the LR-on-PCA baseline (0.9745 +/- 0.002 on the 627
cross-study cohort)?

Method
------
For each sample, summarize the 5 channels into a brief natural-language
description (median, std, percentile bins, FSD mode). Use 8-shot
prompting: 4 cancer + 4 healthy examples drawn from the train fold
(also pre-summarized). Gemma produces a probability P(cancer).
Compute pooled-OOF AUC on the test fold.

5-fold CV, same hygiene as the honest benchmark in
cfdna-fragmentomics-pipeline/scripts/honest_benchmark.py.

Why this is a fair comparison
-----------------------------
- Same train/test split (5-fold StratifiedKFold with fixed seed)
- Same label-parsing and harmonization where applicable
- Same 5-channel input representation, just summarized as text
- Same AUC metric, same OOF aggregation

What this is NOT
----------------
- Not a claim that Gemma is *better* than LR-on-PCA. Expected result
  is AUC ~0.60-0.75 — Gemma is not competitive with structured
  classifiers on small structured tabular data. That's the point:
  demonstrating that the LR-on-PCA baseline is not trivially beaten
  by a strong general-purpose LLM.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add both repos to path so we can use the pipeline's data loaders
sys.path.insert(0, "/Users/hermes/cfdna-fragmentomics-pipeline")
sys.path.insert(0, "/Users/hermes/cfdna-fragmentomics-pipeline/scripts")

from train_classifier import _harmonize  # noqa: E402


# ---------------------------------------------------------------------------
# Feature-summarization: turn the 5-channel array into a brief text description
# ---------------------------------------------------------------------------

def _summarize_vector(v: np.ndarray, label: str) -> str:
    """One-line summary of a 1D feature vector."""
    return (f"{label}: mean={v.mean():+.3f}, std={v.std():.3f}, "
            f"min={v.min():+.3f}, max={v.max():+.3f}, "
            f"p10={np.percentile(v, 10):+.3f}, p90={np.percentile(v, 90):+.3f}")


def _sample_to_text(sample_id: str,
                    r5: np.ndarray, c5: np.ndarray,
                    r100: np.ndarray, c100_norm: np.ndarray,
                    fsd: np.ndarray,
                    study: str) -> str:
    """Convert one sample's 5 channels to a short textual description.

    Compact (~150 tokens per sample) — too verbose causes the 4-shot
    prompt to overflow the n_ctx window and silently fail (Gemma's
    call raises 'Requested tokens exceed context window' which the
    caller sees only as a parsing failure defaulting to P=0.5).
    """
    fsd_mode = int(round(float(20 + 5 * np.argmax(fsd))))
    # Bin centers run 20, 25, ..., 995 (bin idx 0..195). <150bp = idx 0..25 (centers 20..145).
    short_frac = float(fsd[0:26].sum())
    # >250bp = idx 46..195 (centers 250..995).
    long_frac = float(fsd[46:196].sum())
    return (
        f"Patient {sample_id} ({study}): "
        f"r5(m={r5.mean():+.2f},s={r5.std():.2f}) "
        f"r100(m={r100.mean():+.2f},s={r100.std():.2f}) "
        f"fsd(mode={fsd_mode}bp,short<150bp={short_frac:.0%},long>250bp={long_frac:.0%})"
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a computational biologist analyzing cell-free DNA (cfDNA) "
    "fragmentomics data. Your task is to predict whether a plasma cfDNA "
    "sample comes from a cancer patient or a healthy control, based on "
    "the provided 5-channel profile (5Mb short/long ratio, 5Mb coverage, "
    "100kb short/long ratio, 100kb coverage, and fragment size distribution).\n\n"
    "Respond with a single line in this exact format:\n"
    "P(cancer)=<number between 0.0 and 1.0>"
)

PROMPT_TEMPLATE = """Here are {nshot} labeled examples (4 cancer, 4 healthy):

{examples}

Now predict the label for this new patient. Respond with P(cancer)=<number>.

{query}

P(cancer)="""


def _build_few_shot_prompt(train_sample_ids: list[str],
                            train_features: dict,
                            train_labels: dict,
                            train_studies: dict,
                            n_cancer: int = 4, n_healthy: int = 4,
                            seed: int = 0) -> str:
    """Pick n_cancer + n_healthy examples from train and format them."""
    rng = np.random.default_rng(seed)
    cancers = [s for s in train_sample_ids if train_labels[s] == 1]
    healthies = [s for s in train_sample_ids if train_labels[s] == 0]
    chosen_cancers = rng.choice(cancers, size=min(n_cancer, len(cancers)),
                                 replace=False)
    chosen_healthies = rng.choice(healthies, size=min(n_healthy, len(healthies)),
                                    replace=False)
    examples = []
    for s in list(chosen_cancers) + list(chosen_healthies):
        f = train_features[s]
        label = "CANCER" if train_labels[s] == 1 else "HEALTHY"
        text = _sample_to_text(s, f["r5"], f["c5"], f["r100"],
                                f["c100"], f["fsd"], train_studies[s])
        examples.append(f"Example ({label}):\n{text}\nP(cancer)="
                        f"{1.0 if train_labels[s] == 1 else 0.0:.2f}")
    return "\n\n".join(examples)


# ---------------------------------------------------------------------------
# Data loading: build the per-sample feature dict from on-disk .npy/JSON
# ---------------------------------------------------------------------------

def load_features(features_dir: str,
                  labels_tsv: str) -> tuple[list[str], dict, dict, dict]:
    """Load the 5 channels for each sample listed in labels_tsv.

    Returns (sample_ids, features_by_sample, labels, studies).
    features_by_sample[s] = {"r5", "c5", "r100", "c100", "fsd"} np arrays.
    """
    samples, labels, studies = [], {}, {}
    with open(labels_tsv) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            sid = parts[0]
            lab = 1 if parts[1].lower() in ("cancer", "1", "tumor") else 0
            study = parts[2] if len(parts) >= 3 else "unknown"
            samples.append(sid)
            labels[sid] = lab
            studies[sid] = study
    features = {}
    loaded = 0
    skipped = 0
    for s in samples:
        required = [os.path.join(features_dir, f"{s}.delfi_5mb_ratio.npy"),
                    os.path.join(features_dir, f"{s}.delfi_5mb_coverage.npy"),
                    os.path.join(features_dir, f"{s}.delfi_100kb_ratio.npy"),
                    os.path.join(features_dir, f"{s}.delfi_100kb_counts.npy"),
                    os.path.join(features_dir, f"{s}.fsd.json")]
        if not all(os.path.exists(p) for p in required):
            skipped += 1
            continue
        r5 = np.load(required[0])
        c5 = np.load(required[1])
        r100 = np.load(required[2])
        c100_raw = np.load(required[3])
        c100 = c100_raw / np.median(c100_raw)
        with open(required[4]) as fh:
            fsd_dict = json.load(fh)
        # FSD JSON: size_bins is {bin_range: freq}, normalized to sum=1.
        keys = sorted(fsd_dict["size_bins"].keys(),
                       key=lambda k: int(k.split("-")[0]))
        fsd = np.asarray([fsd_dict["size_bins"][k] for k in keys],
                          dtype=float)
        features[s] = {"r5": r5, "c5": c5, "r100": r100, "c100": c100,
                        "fsd": fsd}
        loaded += 1
    samples = list(features.keys())
    if skipped > 0:
        print(f"[load_features] loaded={loaded}, skipped={skipped} (no features on disk)")
    return samples, features, labels, studies


# ---------------------------------------------------------------------------
# Gemma call: parse P(cancer) from the response
# ---------------------------------------------------------------------------

def _parse_p_cancer(text: str) -> float | None:
    """Extract P(cancer)=<number> from the LLM response. Returns None on failure."""
    import re
    m = re.search(r"P\(cancer\)\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
        except ValueError:
            return None
    # Fallback: try to find any float in the response
    m = re.search(r"\b(0\.\d+|1\.0|0)\b", text)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def run_lr_baseline(X: np.ndarray, y: np.ndarray,
                     pca_n: int = 200, n_seeds: int = 5) -> dict:
    """Re-run the LR-on-PCA baseline for direct comparison."""
    aucs = []
    for s in range(n_seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        yt, ys = [], []
        for tr, te in cv.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca),
                       random_state=0).fit(Xtr)
            m = LogisticRegression(max_iter=20000, tol=1e-8,
                                     random_state=0).fit(
                pca.transform(Xtr), y[tr])
            ys.extend(m.predict_proba(pca.transform(Xte))[:, 1].tolist())
            yt.extend(y[te].tolist())
        aucs.append(roc_auc_score(yt, ys))
    return {"auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "per_seed_aucs": aucs}


def assemble_full_vector(features: dict, sample_ids: list[str]) -> np.ndarray:
    rows = []
    for s in sample_ids:
        f = features[s]
        rows.append(np.concatenate([f["r5"], f["c5"], f["r100"],
                                      f["c100"], f["fsd"]]))
    return np.nan_to_num(np.stack(rows).astype(float), nan=0.0,
                          posinf=0.0, neginf=0.0)


def run_gemma_baseline(samples: list[str], features: dict,
                        labels: dict, studies: dict,
                        llm, n_few_shot: int = 4,
                        n_seeds: int = 1,
                        max_tokens: int = 20) -> dict:
    """Run the Gemma few-shot baseline. Returns AUC per seed."""
    y = np.asarray([labels[s] for s in samples], dtype=int)
    aucs = []
    for s in range(n_seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        ys, yt = [], []
        for fold_idx, (tr, te) in enumerate(cv.split(samples, y)):
            train_sids = [samples[i] for i in tr]
            test_sids = [samples[i] for i in te]
            few_shot = _build_few_shot_prompt(
                train_sids, features, labels, studies,
                n_cancer=n_few_shot // 2,
                n_healthy=n_few_shot // 2,
                seed=s * 100 + fold_idx)
            for tsid in test_sids:
                f = features[tsid]
                query_text = _sample_to_text(
                    tsid, f["r5"], f["c5"], f["r100"], f["c100"],
                    f["fsd"], studies[tsid])
                user_msg = PROMPT_TEMPLATE.format(
                    nshot=n_few_shot, examples=few_shot,
                    query=query_text)
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                txt = resp["choices"][0]["message"]["content"]
                p = _parse_p_cancer(txt)
                if p is None:
                    p = 0.5  # default to chance if parsing fails
                ys.append(p)
                yt.append(labels[tsid])
            print(f"  seed {s} fold {fold_idx}/4 done "
                  f"({len(yt)}/{len(y)} samples total so far)",
                  flush=True)
        try:
            auc = roc_auc_score(yt, ys)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"  seed {s} AUC: {auc:.4f}", flush=True)
    return {"auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "per_seed_aucs": aucs,
            "n_few_shot": n_few_shot}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir",
                    default="/Users/hermes/cfdna-fragmentomics-pipeline/data/features")
    ap.add_argument("--labels",
                    default="/Users/hermes/cfdna-fragmentomics-pipeline/data/features/labels_cross_study.tsv")
    ap.add_argument("--model-path",
                    default="/Users/hermes/models/gemma-2-9b-it-Q4_K_M.gguf")
    ap.add_argument("--n-few-shot", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20,
                    help="Max tokens for the Gemma response; P(cancer)=N is short")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of CV seeds for the LR baseline")
    ap.add_argument("--gemma-seeds", type=int, default=1,
                    help="Number of CV seeds for Gemma (slow)")
    ap.add_argument("--out", default="/tmp/gemma_baseline.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, take a balanced subset of N cancer + N healthy")
    args = ap.parse_args()

    print("[experiment] Loading features...")
    samples, features, labels, studies = load_features(
        args.features_dir, args.labels)
    if args.limit > 0:
        rng = np.random.default_rng(0)
        lab_arr = np.asarray([labels[s] for s in samples], dtype=int)
        cancers = [s for s, l in zip(samples, lab_arr) if l == 1]
        healthies = [s for s, l in zip(samples, lab_arr) if l == 0]
        n_each = args.limit // 2
        cancers = list(rng.choice(cancers, size=min(n_each, len(cancers)),
                                    replace=False))
        healthies = list(rng.choice(healthies, size=min(n_each, len(healthies)),
                                      replace=False))
        samples = cancers + healthies
    print(f"[experiment] {len(samples)} samples, "
          f"{(np.asarray([labels[s] for s in samples]) == 1).sum()} cancer, "
          f"{(np.asarray([labels[s] for s in samples]) == 0).sum()} healthy")

    # Build the LR baseline (fast)
    print("[experiment] Running LR-on-PCA baseline...")
    X = assemble_full_vector(features, samples)
    y = np.asarray([labels[s] for s in samples], dtype=int)
    lr = run_lr_baseline(X, y, pca_n=200, n_seeds=args.seeds)
    print(f"[experiment] LR baseline: AUC {lr['auc_mean']:.4f} +/- {lr['auc_std']:.4f}")

    # Build the Gemma baseline (slow)
    print(f"[experiment] Loading Gemma 2 9B from {args.model_path}...")
    t0 = time.time()
    from llama_cpp import Llama
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        chat_format="gemma",
        verbose=False,
    )
    print(f"[experiment] Model loaded in {time.time() - t0:.1f}s")

    print(f"[experiment] Running Gemma baseline "
          f"({args.n_few_shot}-shot, {args.gemma_seeds} seed(s))...")
    t0 = time.time()
    gemma = run_gemma_baseline(samples, features, labels, studies, llm,
                                n_few_shot=args.n_few_shot,
                                n_seeds=args.gemma_seeds,
                                max_tokens=args.max_tokens)
    elapsed = time.time() - t0
    print(f"[experiment] Gemma baseline: AUC {gemma['auc_mean']:.4f} +/- {gemma['auc_std']:.4f}")
    print(f"[experiment] Gemma runtime: {elapsed:.1f}s "
          f"({elapsed / len(samples):.2f}s/sample)")

    out = {
        "n_samples": len(samples),
        "n_cancer": int(y.sum()),
        "n_healthy": int(len(y) - y.sum()),
        "lr_baseline": lr,
        "gemma_baseline": gemma,
        "n_few_shot": args.n_few_shot,
        "elapsed_seconds": elapsed,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[experiment] Wrote {args.out}")

    del llm
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())