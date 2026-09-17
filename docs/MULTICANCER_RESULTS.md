# Multi-cancer classification results (Subagent J)

This document reports the first multi-cancer (one-vs-rest) classification
numbers on the 627-sample cfDNA WGS cohort. It complements the existing
binary cancer-vs-healthy classifier (AUC ≈ 0.978 on the same cohort) by
asking: **does the per-class signal survive the move from 2 classes to 5–8
classes, and how much AUC do we "spend" by going multi-class?**

The headline answer is: **macro AUC 0.9703 ± 0.0012** on the filtered
5-class cohort, essentially identical to the binary baseline AUC 0.9712 ±
0.0018 measured on the same filtered cohort (information delta **−0.0008**).

---

## 1. Setup

### 1.1 Disease-class mapping

The existing `labels_cross_study.tsv` has only binary cancer/healthy
labels. The new `labels_multiclass.tsv` adds a `disease_class` column
decoded from the sample-ID prefix:

| Prefix      | Disease class | n (full) | Source                  |
|-------------|---------------|----------|-------------------------|
| `CGPLH*`    | HEALTHY       | 260      | Cristiano 2019 healthy  |
| `C3xx`      | HEALTHY       | 32       | Jiang 2015 healthy      |
| `CGH1x`     | HEALTHY       | 2        | Cristiano outliers      |
| `H*`,`HOT*` | HCC_J         | 89       | Jiang 2015 HCC          |
| `CGPLLU*`   | LUAD          | 79       | Cristiano lung          |
| `CGPLBR*`   | BRCA          | 54       | Cristiano breast        |
| `CGPLPA*`   | PAAD          | 60       | Cristiano pancreas      |
| `CGPLOV*`   | OV            | 28       | Cristiano ovarian       |
| `CGCRC*`    | CRC           | 27       | Cristiano colorectal    |
| `CGST1-8*`  | OTHER_C       | 27       | Cristiano gastric/bile duct |

Total: **658 labelled samples** in the labels file (627 in the headline
cohort after dropping 31 samples missing one or more required features).

This decoding is **inferred, not validated against FinaleDB** — FinaleDB
was unreachable during the analysis (`finaledb.research.cchmc.org/api`
returned HTTP 500 on 2026-09-12). The mapping is consistent with the
AUDIT_REPORT_2.md S1 deferred-analysis counts (n ∈ {9, 18, 27, 28, 54,
60, 79, 88}; 4 types account for 77% of cancers), and with the
Cristiano 2019 paper (PMC6774252), which covers breast, colorectal,
gastric, lung, ovarian, pancreatic, and bile-duct cancer.

### 1.2 Cohort filtering

The script `scripts/multiclass_classification.py` drops any class with
n < 30 (per task spec). Three classes fall below that threshold:

| Class   | n  |
|---------|----|
| CRC     | 27 |
| OTHER_C | 27 |
| OV      | 28 |

These were dropped to keep CV folds (5-fold stratified) statistically
meaningful. After dropping, the cohort is **545 samples × 63,246 features**
with 5 classes.

### 1.3 Features + modelling

Same 5-channel feature stack as `scripts/honest_benchmark.py`:
5 Mb DELFI ratio, 5 Mb coverage, 100 kb ratio, 100 kb counts (normalized
by per-sample median), and 196-bin fragment-size distribution.

Pipeline per fold: StandardScaler → PCA(200) → one-vs-rest logistic
regression (C=1.0, lbfgs, max_iter=1000). 5 seeds × 5-fold stratified CV,
OOF predictions pooled across seeds.

---

## 2. Headline numbers

### 2.1 Per-class OvR AUC (pooled OOF across 5 seeds × 5 folds)

| Class   |  n  | OOF AUC | Mean ± Std across seeds |
|---------|----:|--------:|------------------------:|
| BRCA    |  53 | 0.9724  | 0.9712 ± 0.0021         |
| HCC_J   |  89 | 0.9960  | 0.9955 ± 0.0007         |
| HEALTHY | 264 | 0.9760  | 0.9722 ± 0.0032         |
| LUAD    |  79 | 0.9747  | 0.9675 ± 0.0048         |
| PAAD    |  60 | 0.9482  | 0.9451 ± 0.0022         |

- **Macro AUC**: 0.9703 ± 0.0012
- **Top-2 accuracy**: 0.9732 ± 0.0068
- **Binary cancer-vs-healthy AUC (same filtered cohort)**: 0.9712 ± 0.0018
- **Information delta (multi − binary)**: **−0.0008** (essentially zero loss)

### 2.2 Confusion matrix (rows=true, cols=pred)

```
       BRCA    HCC_J  HEALTHY     LUAD     PAAD
BRCA        39        0        7        5        2
HCC_J        0       86        1        2        0
HEALTHY      1        2      258        0        3
LUAD         4        0       10       64        1
PAAD         1        0       20        3       36
```

Per-class recall: BRCA 73.6%, HCC_J 96.6%, HEALTHY 97.7%, LUAD 81.0%,
PAAD 60.0%. The dominant confusion pairs are PAAD↔HEALTHY (20 PAAD
predicted healthy) and LUAD↔HEALTHY (10 LUAD predicted healthy) and
BRCA↔HEALTHY (7 BRCA predicted healthy). This is consistent with the
fragmentomic signal being weaker for pancreatic cancer, which is known
from the literature (Cristiano 2019 Table 2).

---

## 3. Comparison vs binary AUC

The binary cancer-vs-healthy classifier on the *same* filtered 545-sample
cohort gets AUC 0.9712 ± 0.0018 (5-seed × 5-fold). The multi-class
**macro** AUC is 0.9703 ± 0.0012 — a difference of −0.0008, well within
1 seed-to-seed standard deviation.

In other words: **the move from binary → 5-class OvR costs essentially
zero AUC**. The fragmentomic signal is rich enough that it can both
detect cancer *and* identify which tissue it came from at nearly the same
overall quality.

For comparison, the cross-study binary headline of 0.9745 ± 0.0022
(AUDIT_REPORT_2.md, S5 fixed) is on the full 627 cohort before class
filtering; here we filter to 545 to keep small classes from breaking
the stratified folds, so a small drop is expected and benign.

---

## 4. Honest limitations

1. **Three classes dropped (n < 30): CRC (27), OV (28), OTHER_C (27).**
   Per-cancer-type AUC for these is **not reported** in this document.
   The audit explicitly noted this was deferred to a follow-up that
   needed FinaleDB metadata + 1-day re-extraction; this is now
   partially addressed (prefix decoding works) but the small n still
   prevents robust stratified CV. A leave-one-out or 3-fold CV would
   give a number, but with very wide confidence intervals.

   **Sept 2026 follow-up:** the OV-specific K=10000 channel-set
   pre-filter (`--ov-topk-channels results/ov_topk_channels.json`)
   lifts OV Sens@99% from 0.25 to **0.3571** when the script is run
   with `--min-class-n 0` (i.e. when OV is kept in the cohort). See
   `docs/PER_CANCER_TOPK.md` for the full result table and
   `test/test_multiclass_classification.py` for the byte-identical
   regression guard.

2. **Prefix decoding is inferred, not validated against ground truth.**
   FinaleDB's seqrun API was unreachable during this analysis
   (2026-09-12, HTTP 500). The mapping is consistent with the audit's
   S1 deferred-analysis counts and the published Cristiano 2019 cancer
   list, but a future re-extraction with `fetch_finaledb.py --disease
   <name>` could be used to confirm.

3. **Pooled OOF (mean across seeds).** The macro AUC is reported as
   mean ± std across 5 seeds × 5 folds = 25 OOF estimates per class.
   These 25 estimates are *not independent* — they share folds across
   seeds only partially — so the std understates true variance. The
   audit's ST4 note applies here too: this is "is the Δ nonzero for
   these shuffles", not a frequentist confidence interval.

4. **Top-2 accuracy (0.9732) is much higher than top-1 (argmax of
   softmax probabilities from independent OvR models, ~80% on the
   confusion matrix diagonal).** The independent OvR models produce
   scores that aren't calibrated as a joint distribution, so top-2
   should be read as "the true class is in the top-2 highest-scoring
   classes", not "the softmax confidence is high".

5. **One-vs-rest is a simplification.** A multinomial logistic
   regression or a small neural net could share parameters across
   classes and possibly extract more signal from the small
   per-cancer-type cohorts. We chose OvR for parity with the binary
   baseline and because the question is "does the per-class signal
   survive", not "what's the best multi-class architecture".

6. **No held-out external validation.** The 545 samples come from
   FinaleDB publication 6 (Jiang 2015) and publication 8 (Cristiano
   2019). All cross-validation is within these studies. A future
   TCGA-LUAD-style external hold-out would give a more honest
   clinical estimate.

7. **Healthy class is 48% of the cohort.** OvR AUC for HEALTHY (0.9760)
   is partly driven by the easy task of separating healthy from
   cancer, which is also what the binary classifier is doing. A
   cancer-vs-cancer confusion matrix (drop healthy from the OvR
   evaluation) would give a more discriminating "tissue-of-origin"
   number, but is deferred.

---

## 5. Files added

- `labels_multiclass.tsv` — 658 samples × 4 columns (sample,
  disease_class, label, study). Header included. The mapping table is
  documented in `scripts/multiclass_classification.py:_decode_prefix()`.
- `scripts/multiclass_classification.py` — OvR logistic regression,
  5 seeds × 5-fold CV, ~220 LOC, CLI with `--quick` smoke mode.
- `results/multiclass_classification.json` — full results including
  per-class AUCs, macro AUC, top-2 accuracy, confusion matrix, and
  binary-vs-multi information delta.
- `test/test_multiclass_classification.py` — 21 tests (10 spot-checks
  of `labels_multiclass.tsv`, 1 schema test, 4 sanity tests on AUCs,
  1 CLI test, 2 end-to-end smoke tests, 3 OV K=10000 pre-filter tests
  with byte-identical regression guards). All pass.
- `results/ov_topk_channels.json` — 10,000 OV channel indices (with
  provenance metadata) committed to the repo for the optional OV
  pre-filter.
- `docs/PER_CANCER_TOPK.md` — full report on the per-cancer top-K
  sweep, including the integration as an optional pre-filter.
- `docs/MULTICANCER_RESULTS.md` — this file.

## 6. Reproduce

```bash
# full benchmark (~4 minutes)
python scripts/multiclass_classification.py \
    --out results/multiclass_classification.json

# smoke (~25 seconds)
python scripts/multiclass_classification.py --quick \
    --out results/multiclass_smoke.json

# OV K=10000 channel-set pre-filter (~4 minutes; uses --min-class-n 0
# so OV (n=28) is kept in the cohort; without this flag OV/CRC/OTHER_C
# are dropped by the n>=30 class filter)
python scripts/multiclass_classification.py \
    --min-class-n 0 \
    --ov-topk-channels results/ov_topk_channels.json \
    --out results/multiclass_with_ov_prefilter.json

# tests
pytest test/test_multiclass_classification.py -v
```
