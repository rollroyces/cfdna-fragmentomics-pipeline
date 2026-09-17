# Comprehensive Sens@Spec + PPV@Prev table

Generated: 2026-09-17T08:02:52.894603+00:00

Cohort: 627 samples (363 cancer + 264 healthy), 5-seed × 5-fold pooled OOF.

---

# Sens@Spec table — frag-only vs frag+fusion

Pooled 5-seed × 5-fold OOF on the 627-sample cross-study cohort.
Frag-only = 5-channel LR + PCA(200) on FinaleDB features.
Frag+fusion = naive average of frag-only score and a synthetic
mutation-channel score calibrated to AUC=0.92, Sens@95%≈0.77
(same protocol as `scripts/fusion_ablation.py`).

CI: **DeLong** placement-value method (DeLong et al. 1988).
For comparison, percentile-bootstrap CIs (n=1000) are also shown
where available.

**Pooled AUC:** frag-only = **0.9780**, frag+fusion = **0.9921**.

**Operating-point sens for PPV table:** sens @ spec = 99.0%.

| spec | frag-only sens | DeLong 95% CI | bootstrap 95% CI | frag+fusion sens | DeLong 95% CI | bootstrap 95% CI |
|---:|---:|---|---|---:|---|---|
| 90.00% | 94.63% | [92.32, 96.93] | [89.35, 97.80] | 98.48% | [97.26, 99.71] | [96.55, 99.73] |
| 95.00% | 89.67% | [86.55, 92.79] | [84.28, 93.21] | 93.80% | [91.33, 96.27] | [90.19, 98.66] |
| 98.00% | 85.26% | [81.62, 88.90] | [73.84, 89.57] | 91.32% | [88.44, 94.21] | [83.20, 94.60] |
| 99.00% | 75.34% | [70.91, 79.78] | [63.74, 87.70] | 84.16% | [80.41, 87.91] | [79.89, 93.61] |
| 99.50% | 75.34% | [70.91, 79.78] | [61.34, 86.23] | 83.88% | [80.11, 87.66] | [78.53, 92.21] |
| 99.90% | 64.60% | [59.68, 69.52] | [60.45, 80.35] | 81.13% | [77.11, 85.15] | [78.10, 89.49] |

---

# PPV@Prev table — Sens@99.0% spec operating point

Operating point: spec = **99.0%**.

Pooled sens at this operating point (frag-only): **75.34%** (DeLong 95% CI [70.91, 79.78]).

Pooled sens at this operating point (frag+fusion): **84.16%** (DeLong 95% CI [80.41, 87.91]).

| prevalence | context | frag-only PPV | frag+fusion PPV |
|---:|---|---|---|
| 0.10% | 0.1% — ultra-low-prevalence research cohort | 7.01% | 7.77% |
| 0.40% | 0.4% — US adults 50+ (Galleri-comparable) | 23.23% | 25.26% |
| 1.00% | 1% — adults 50+ with 1 risk factor | 43.22% | 45.95% |
| 5.00% | 5% — high-risk surveillance (post-MRD) | 79.86% | 81.58% |
| 10.00% | 10% — hereditary cancer syndrome surveillance | 89.33% | 90.34% |
| 20.00% | 20% — symptomatic workup | 94.96% | 95.46% |
| 50.00% | 50% — confirmatory test | 98.69% | 98.83% |

---

# Per-cancer Sens@Spec table (OvR pooled 5-seed × 5-fold OOF)

Each row: one-vs-rest pooled OOF predictions for that cancer vs (all other cancers + healthy). CI = DeLong placement-value.
Bootstrap AUC CI is preserved in the JSON (cross-check from `results/per_cancer_auc.json`).

| cancer | n_pos | pooled AUC | DeLong 95% CI | sens@90.0% | DeLong 95% CI | sens@95.0% | DeLong 95% CI | sens@98.0% | DeLong 95% CI | sens@99.0% | DeLong 95% CI | sens@99.5% | DeLong 95% CI | sens@99.9% | DeLong 95% CI |
|---|---:|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|
| HCC_J | 89 | 0.9969 | [0.9942, 0.9996] | 99.44% | [98.34, 100.00] | 96.07% | [92.16, 99.97] | 93.82% | [88.91, 98.73] | 93.82% | [88.91, 98.73] | 92.70% | [87.37, 98.02] | 75.84% | [66.97, 84.72] |
| CRC <30 | 27 | 0.9743 | [0.9551, 0.9935] | 90.74% | [80.23, 100.00] | 87.04% | [74.67, 99.41] | 68.52% | [51.05, 85.98] | 50.00% | [31.14, 68.86] | 50.00% | [31.14, 68.86] | 9.26% | [0.00, 19.77] |
| HEALTHY | 264 | 0.9698 | [0.9577, 0.9819] | 93.37% | [90.39, 96.35] | 85.80% | [81.59, 90.00] | 56.25% | [50.27, 62.23] | 32.01% | [26.38, 37.63] | 16.10% | [11.67, 20.52] | 16.10% | [11.67, 20.52] |
| LUAD | 79 | 0.9607 | [0.9375, 0.9838] | 90.51% | [84.12, 96.89] | 85.44% | [77.72, 93.17] | 60.13% | [49.33, 70.92] | 50.00% | [38.97, 61.03] | 24.68% | [15.20, 34.17] | 0.00% | [0.00, 0.00] |
| OV <30 | 28 | 0.9546 | [0.9318, 0.9774] | 83.93% | [70.54, 97.32] | 69.64% | [52.67, 86.62] | 30.36% | [13.38, 47.33] | 23.21% | [7.69, 38.74] | 5.36% | [0.00, 13.07] | 5.36% | [0.00, 13.07] |
| OTHER_C <30 | 27 | 0.9541 | [0.9209, 0.9873] | 90.74% | [80.23, 100.00] | 68.52% | [51.05, 85.98] | 61.11% | [42.74, 79.48] | 38.89% | [20.52, 57.26] | 31.48% | [14.02, 48.95] | 12.96% | [0.59, 25.33] |
| BRCA | 53 | 0.9532 | [0.9318, 0.9747] | 85.85% | [76.56, 95.14] | 66.98% | [54.34, 79.63] | 50.00% | [36.54, 63.46] | 40.57% | [27.35, 53.78] | 38.68% | [25.57, 51.78] | 17.92% | [7.67, 28.18] |
| PAAD | 60 | 0.9401 | [0.9059, 0.9743] | 87.50% | [79.22, 95.78] | 79.17% | [68.94, 89.40] | 55.83% | [43.27, 68.40] | 44.17% | [31.60, 56.73] | 34.17% | [22.18, 46.16] | 9.17% | [1.99, 16.34] |

---

## Honest framing

* All numbers are pooled out-of-fold on the SAME 627-sample
  FinaleDB cohort. They are NOT external validation.
* CI: DeLong placement-value method on a fixed-threshold operating
  point; bootstrap CI is percentile-resampled (n=1000) for
  sanity-check comparison.
* Frag+fusion uses a synthetic mutation score calibrated to
  AUC=0.92, Sens@95%≈0.77 (same recipe as
  `scripts/fusion_ablation.py`). The mutation channel is a
  proxy, not a real targeted-panel readout.
* Per-cancer rows with n<30 are exploratory (flagged in column).
