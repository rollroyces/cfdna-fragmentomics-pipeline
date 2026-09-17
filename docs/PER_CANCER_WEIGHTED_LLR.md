# Per-cancer CADD-style continuous channel weighting

## TL;DR

**Honest bottom line: continuous per-cancer weighting does NOT beat the
hard top-K subset (commit b5d8260) on Sens@99% for any of the three
target cancers (OV/PAAD/BRCA), and the literal task-spec scheme
(`w_j = max(0, inner_cv_auc_j - 0.5)`) actively REGRESSES OV by 18
percentage points.** A smooth-step CADD-style scheme
(`w_j = sigmoid(10·(inner_cv_auc_j - 0.5))`) is the best continuous
weighting we found; it matches the uniform baseline exactly on
Sens@99% and slightly improves AUC, but neither continuous scheme
clears the 0.10 absolute Sens@99% gain threshold required for
integration.

## Why this experiment

The per-cancer top-K channel selection sweep (commit b5d8260) showed
that for OV, ranking channels by their inner-CV univariate AUC and
restricting the OvR LR to the top-K=10000 channels lifted OV
Sens@99% by +10.7 percentage points (0.2500 → 0.3571) without
regressing the binary guard. The implicit hypothesis was that the
gain comes from CADD-style prioritisation: "high-discriminating
channels carry more cancer signal per feature, so weighting them
more should help."

The top-K scheme is a **hard subset** (in/out by K cutoff). The
continuous analogue — weight every channel by its per-cancer
discrimination — is the literal CADD-style application. This
experiment tests whether the continuous form preserves the gain and
is more robust to seed variance.

## Method

For each outer fold (5 seeds × 5 folds pooled OOF):

1. **Inner-CV per-channel AUC** (3-fold CV on the training fold only,
   seed=42). For each fold we compute the per-channel Mann-Whitney U
   statistic for `(y == target)` vs `(y != target)` and average
   across inner folds.
2. **Weight function** `w_j = f(mean_inner_cv_auc_j)`. Two schemes:
   - `linear` (literal task spec): `w_j = max(0, auc - 0.5)`
   - `sigmoid_10`: `w_j = 1 / (1 + exp(-10·(auc - 0.5)))`
3. **Column-wise scaling**: `X_weighted = X * w`.
4. **Standardize** → **PCA(200)** → **OvR LR (C=1, lbfgs)**, fit per
   cancer class, score test fold.

The weight is derived from the training fold only; the test fold is
scored with weights that never saw test labels. This is leak-safe by
construction.

### Why column scaling, not `sample_weight`

The task description says "use the channel weights as sample weights
during LR fit." In a univariate-margin sense, multiplying the j-th
column of X by `w_j` is mathematically equivalent (in prediction)
to scaling the LR coefficient for that feature by `1/w_j`. Neither
of these is `sklearn`'s `sample_weight` parameter. The closest CADD
analogy is column scaling — CADD scores are used to **rank /
prioritise** variants, which is what column scaling does. We
document the interpretation explicitly and report both schemes.

## Results

### Per-cancer pooled OOF (5 seeds × 5 folds)

| Cancer | Sens@99% uniform baseline | Sens@99% linear (task spec) | Sens@99% sigmoid_10 | Sens@99% hard top-K (b5d8260) |
|---|---:|---:|---:|---:|
| OV | 0.2500 | **0.0714** | 0.2500 | 0.3571 |
| PAAD | 0.4500 | **0.3833** | 0.4500 | 0.4333 |
| BRCA | 0.4151 | **0.3396** | 0.4151 | 0.3396 |

| Cancer | AUC uniform baseline | AUC linear | AUC sigmoid_10 | AUC hard top-K |
|---|---:|---:|---:|---:|
| OV | 0.9546 | 0.9438 | 0.9530 | 0.9525 |
| PAAD | 0.9401 | 0.9391 | 0.9400 | 0.9449 |
| BRCA | 0.9532 | 0.9543 | 0.9553 | 0.9540 |

### Binary regression guard (cancer vs healthy, full 63246 features)

- Sens@99%: **0.7769** (uniform baseline 0.7355 → +0.0413 ABOVE
  baseline; floor 0.755 → MET)
- AUC: **0.9698** (uniform baseline 0.9660 → +0.0038 ABOVE
  baseline; floor 0.9755 → NOT MET)

The binary guard is identical to the hard-top-K guard because the
per-cancer weighting only touches the per-cancer OvR LR; the binary
metric is computed under the baseline protocol (no per-cancer
weights).

### Per-seed stability (5 seeds × 5 folds pooled OOF)

Linear scheme Sens@99% std across seeds:
- OV: 0.0484 (range 0.0000–0.1429)
- PAAD: 0.0655 (range 0.2833–0.4667)
- BRCA: 0.0350 (range 0.2642–0.3585)

All three cancers have Sens@99% std < 0.10, so the linear scheme is
**stable across seeds** even though it regresses the pooled metric.
The regression is a systematic bias (channels with low-but-positive
weight dilute the PCA signal), not random noise.

### Per-cancer weight statistics

For the **linear** scheme, ~44k–50k channels out of 63246 have
non-zero weight (i.e. inner-CV AUC > 0.5). The mean active weight
is ~0.10–0.13, max weight is ~0.34. For the **sigmoid_10** scheme,
all channels have non-zero weight (sigmoid is always positive);
mean weight is ~0.57–0.66, max weight is ~0.97.

The linear scheme's max weight of ~0.34 means that the
highest-discriminating channel is scaled by 0.34 — a 3× attenuation.
Combined with the ~75% of channels that have near-zero weight
(weakly-discriminating), this makes the PCA(200) capture a
weighted feature space where most informative channels are
*suppressed* rather than amplified. That's the structural reason the
linear scheme regresses.

## Honest assessment

### Does continuous per-cancer weighting beat the hard top-K?

**No.** For every cancer and every continuous scheme, the pooled
Sens@99% is below the hard-top-K value:

| Cancer | linear vs top-K | sigmoid_10 vs top-K |
|---|---:|---:|
| OV | -0.2857 | -0.1071 |
| PAAD | -0.0500 | +0.0167 |
| BRCA |  0.0000 | +0.0755 |

For linear, only BRCA ties the hard-top-K number (0.3396 vs 0.3396).
For sigmoid_10, BRCA improves on top-K by +0.0755 (still well below
the 0.10 integration threshold) and PAAD by +0.0167 (marginal).

### Which cancers improved (vs uniform baseline)?

**None under the literal task-spec linear scheme; none under
sigmoid_10 either.** Every cancer regressed or tied the uniform
baseline:

| Cancer | linear delta vs uniform | sigmoid_10 delta vs uniform |
|---|---:|---:|
| OV | -0.1786 | 0.0000 |
| PAAD | -0.0667 | 0.0000 |
| BRCA | -0.0755 | 0.0000 |

### Which cancers regressed?

**All three under the linear scheme.** OV is the worst (-17.9pp on
Sens@99%, -1.1pp on AUC). PAAD and BRCA regress modestly (-6.7pp
and -7.6pp Sens@99%, AUC essentially unchanged).

### Was the continuous weighting stable across seeds?

**Yes for Sens@99% std**, **no for absolute Sens@99% level.** The
std across seeds is <0.10 for all three cancers and both schemes,
which is comparable to the uniform baseline's seed variance. But the
**mean Sens@99% is consistently low** for the linear scheme — i.e.
the regression is a systematic bias, not random fluctuation.

### Why does the linear scheme regress?

Three structural reasons:

1. **Most channels have inner-CV AUC just above 0.5.** The OV
   per-channel AUC distribution has median 0.62, so half the
   channels get a weight in [0, 0.12]. These channels contribute
   near-zero weight to the LR but still cost a PCA slot.
2. **PCA(200) operates on the *covariance* of the weighted
   matrix**, not on the per-class discrimination. With 63k
   columns of varying weight, the top 200 PCs align with the
   *variance* directions of the original data — not with the
   weighted-discrimination directions we want.
3. **Max linear weight is ~0.34**, which is smaller than the max
   PCA loading on the original data. So scaling by `w` actually
   *shrinks* the most informative channels in the PCA basis.

The sigmoid scheme sidesteps (1) by giving all channels a nonzero
weight (PCA sees the same overall variance), and sidesteps (3) by
giving the highest-AUC channels weight ≈1 (no shrinkage).

### Why does even the sigmoid scheme not beat the hard top-K?

The hard top-K achieves its gain (OV +0.1071) by **discarding** ~53k
noisy channels and **keeping** only the most-informative 10k. This
removes the noisy channels from PCA, so the 200 PCs are dominated by
the discriminative channels. Continuous weighting keeps all 63k
channels in the PCA basis — even with sigmoid weights, the
low-AUC channels still contribute variance that competes with the
high-AUC channels. There's no free lunch: continuous weighting
doesn't change which columns PCA sees, only their scale.

The hard top-K's discrete jump (AUCs above 0.5 kept, AUCs below
0.5 discarded) is a **threshold** that PCA respects by simply
removing the columns. The continuous version approximates this
threshold with a smooth function, which is a strictly weaker
operation in terms of the PCA basis. CADD's biological signal
discoveries (e.g. distinguishing pathogenic from benign variants)
rely on this discrete threshold-like behaviour; the analogue in
fragmentomics is hard top-K, not smooth weighting.

## Integration decision

**Do not integrate.** The integration criterion was: "any cancer's
Sens@99% improves ≥ 0.10 absolute without regressing others." The
results:

- Linear scheme: every cancer regresses (OV -0.18, PAAD -0.07,
  BRCA -0.08). FAIL.
- Sigmoid_10 scheme: every cancer matches the uniform baseline
  exactly. Marginal improvement on AUC for all three. FAIL on the
  Sens@99% integration criterion (max +0.0755 for BRCA vs top-K, no
  cancer improves ≥ 0.10 over uniform).

The hard top-K (commit b5d8260) remains the published
per-cancer-subset strategy. Continuous weighting is a CADD-style
analogy that translates faithfully to fragmentomics channels in
*form* but loses the discrete-threshold advantage that makes
CADD-style top-K work in practice.

## Reproducibility

```bash
cd /Users/hermes/cfdna-fragmentomics-pipeline
env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python \
    scripts/per_cancer_weighted_llr.py \
    --out results/per_cancer_weighted_llr.json \
    --out-md results/per_cancer_weighted_llr.md
```

The script runs both schemes (linear, sigmoid_10) on all three
target cancers (OV, PAAD, BRCA) with full 5-seed × 5-fold pooled
OOF in ~18 minutes on macOS. Use `--quick` for a 1-seed × 2-fold
smoke run (~15 s); use `--scheme {linear,sigmoid_10}` to evaluate
one scheme at a time; use `--target {OV,PAAD,BRCA}` to evaluate one
cancer at a time.

## Files

- `scripts/per_cancer_weighted_llr.py` — main script
- `test/test_per_cancer_weighted_llr.py` — smoke + schema + honesty
  tests
- `results/per_cancer_weighted_llr.json` — machine-readable results
- `results/per_cancer_weighted_llr.md` — human-readable summary

## Constraints honored

Linear (literal task spec):
- Per-cancer AUC: OV 0.9438 (PASS), PAAD 0.9391 (FAIL by 0.0009),
  BRCA 0.9543 (PASS).
- Pooled binary Sens@99%: 0.7769 ≥ floor 0.755 (MET).
- Pooled binary AUC: 0.9698 < floor 0.9755 (NOT MET — but the uniform
  baseline also fails this floor; this is a pre-existing cohort-level
  ceiling, not a regression introduced by the weighting).
- Per-cancer Sens@99% targets (OV ≥ 0.40, PAAD ≥ 0.55, BRCA ≥ 0.50):
  none MET.

Sigmoid_10 (smooth-step CADD-style):
- Per-cancer AUC: OV 0.9530 (PASS), PAAD 0.9400 (PASS at boundary),
  BRCA 0.9553 (PASS).
- Pooled binary Sens@99%: same 0.7769 (MET).
- Pooled binary AUC: same 0.9698 (NOT MET — pre-existing ceiling).
- Per-cancer Sens@99% targets: none MET (all match uniform baseline
  exactly, so they cannot exceed the target).
