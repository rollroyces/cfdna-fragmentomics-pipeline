# Per-cancer OV + PAAD feature-engineering sweep

Generated: 2026-09-16T15:26:27.327371+00:00
Cohort: 627 samples × 63246 features, 5 seeds × 5 folds pooled OOF.

## Baseline (no per-cancer engineering)

| Class | n | OOF AUC | Sens@99% (pooled) |
|---|---:|---:|---:|
| BRCA | 53 | 0.9532 | 0.4151 |
| CRC | 27 | 0.9743 | 0.5185 |
| HCC_J | 89 | 0.9969 | 0.9438 |
| HEALTHY | 264 | 0.9698 | 0.3220 |
| LUAD | 79 | 0.9607 | 0.5063 |
| OTHER_C | 27 | 0.9541 | 0.4074 |
| OV | 28 | 0.9546 | 0.2500 |
| PAAD | 60 | 0.9401 | 0.4500 |

Binary (cancer vs healthy) pooled OOF AUC: 0.9660
Binary Sens@99%: 0.7355  (must not regress below 0.755)

Targets: OV Sens@99% ≥ 0.4, PAAD Sens@99% ≥ 0.6

## Techniques

Each technique's row reports OV/PAAD Sens@99% and AUC, plus a pass/fail against the regression guards (binary Sens@99% ≥ 0.755, OV/PAAD AUC ≥ 0.94).

| Technique | OV AUC | OV Sens@99% | PAAD AUC | PAAD Sens@99% | Binary Sens@99% | OV target met | PAAD target met |
|---|---:|---:|---:|---:|---:|---|---|
| A_score_distribution | 0.9546 | 0.2500 | 0.9401 | 0.4500 | 0.7355 | ❌ | ❌ |
| B_cancer_specific_pca | 0.9484 | 0.3571 | 0.9196 | 0.3333 | 0.7355 | ❌ | ❌ |
| C_threshold_calibration_c2 | 0.5889 | 0.2500 | 0.9279 | 0.4333 | 0.7355 | ❌ | ❌ |
| C_threshold_calibration_c1 | 0.9546 | 0.2500 | 0.9401 | 0.4500 | 0.7355 | ❌ | ❌ |
| D_engineered_features | 0.9528 | 0.2500 | 0.9396 | 0.4500 | 0.7355 | ❌ | ❌ |
| E_class_weight_5x | 0.9520 | 0.2500 | 0.9392 | 0.4667 | 0.7355 | ❌ | ❌ |
| E_class_weight_10x | 0.9531 | 0.2500 | 0.9408 | 0.4667 | 0.7355 | ❌ | ❌ |
| F_margin_features | 0.9546 | 0.2500 | 0.9398 | 0.4500 | 0.7355 | ❌ | ❌ |
| G_B_avg_E5 | 0.9580 | 0.3571 | 0.9346 | 0.4167 | 0.7355 | ❌ | ❌ |
| G_B_avg_E10 | 0.9585 | 0.3571 | 0.9356 | 0.4167 | 0.7355 | ❌ | ❌ |
| G_B_avg_base | 0.9587 | 0.3571 | 0.9353 | 0.4000 | 0.7355 | ❌ | ❌ |
| G_B_avg_E5_avg_E10 | 0.9585 | 0.3214 | 0.9379 | 0.4167 | 0.7355 | ❌ | ❌ |
| G_all_four_avg | 0.9592 | 0.2857 | 0.9394 | 0.4333 | 0.7355 | ❌ | ❌ |

## Per-technique detail

### A_score_distribution

```json
{
  "ov_auc": 0.9545671357023611,
  "ov_sens99": 0.25,
  "paad_auc": 0.9400940623162845,
  "paad_sens99": 0.45,
  "binary_sens99": 0.7355371900826446,
  "ov_dist": {
    "n_pos": 28,
    "n_neg": 599,
    "score_on_positives_mean": 0.16755368978898316,
    "score_on_positives_std": 0.25008815224261793,
    "score_on_positives_skew": 1.6955988495200147,
    "score_on_positives_kurt": 2.086128010939711,
    "score_on_negatives_mean": 0.008465065965512445,
    "score_on_negatives_std": 0.05931436434634279,
    "operating_threshold_at_99pct_spec": 0.35508716394215833,
    "positives_above_threshold": 7
  },
  "paad_dist": {
    "n_pos": 60,
    "n_neg": 567,
    "score_on_positives_mean": 0.4294506090562732,
    "score_on_positives_std": 0.40898177132385766,
    "score_on_positives_skew": 0.3170277107175869,
    "score_on_positives_kurt": -1.6452447496743983,
    "score_on_negatives_mean": 0.011403407395882269,
    "score_on_negatives_std": 0.08086342525609466,
    "operating_threshold_at_99pct_spec": 0.3891327738441972,
    "positives_above_threshold": 27
  }
}
```

### B_cancer_specific_pca

```json
{
  "ov_auc": 0.9483663248270927,
  "ov_sens99": 0.35714285714285715,
  "paad_auc": 0.9195767195767196,
  "paad_sens99": 0.3333333333333333,
  "binary_sens99": 0.7355371900826446,
  "per_target": {
    "OV": {
      "auc_pooled": 0.9483663248270927,
      "sens99_pooled": 0.35714285714285715
    },
    "PAAD": {
      "auc_pooled": 0.9195767195767196,
      "sens99_pooled": 0.3333333333333333
    }
  }
}
```

### C_threshold_calibration_c2

```json
{
  "ov_auc": 0.5888981636060099,
  "ov_sens99": 0.25,
  "paad_auc": 0.9278953556731334,
  "paad_sens99": 0.43333333333333335,
  "binary_sens99": 0.7355371900826446,
  "per_target": {
    "OV": {
      "most_confusable_cancer": "BRCA",
      "s_neg_mean": 0.008465065965512445,
      "c1_sens99": 0.25,
      "c1_auc": 0.9545671357023611,
      "c2_sens99": 0.25,
      "c2_auc": 0.5888981636060099,
      "baseline_ovr_sens99": 0.25,
      "baseline_ovr_auc": 0.9545671357023611
    },
    "PAAD": {
      "most_confusable_cancer": "CRC",
      "s_neg_mean": 0.011403407395882269,
      "c1_sens99": 0.45,
      "c1_auc": 0.9400940623162846,
      "c2_sens99": 0.43333333333333335,
      "c2_auc": 0.9278953556731334,
      "baseline_ovr_sens99": 0.45,
      "baseline_ovr_auc": 0.9400940623162845
    }
  }
}
```

### C_threshold_calibration_c1

```json
{
  "ov_auc": 0.9545671357023611,
  "ov_sens99": 0.25,
  "paad_auc": 0.9400940623162846,
  "paad_sens99": 0.45,
  "binary_sens99": 0.7355371900826446,
  "per_target": {
    "OV": {
      "most_confusable_cancer": "BRCA",
      "s_neg_mean": 0.008465065965512445,
      "c1_sens99": 0.25,
      "c1_auc": 0.9545671357023611,
      "c2_sens99": 0.25,
      "c2_auc": 0.5888981636060099,
      "baseline_ovr_sens99": 0.25,
      "baseline_ovr_auc": 0.9545671357023611
    },
    "PAAD": {
      "most_confusable_cancer": "CRC",
      "s_neg_mean": 0.011403407395882269,
      "c1_sens99": 0.45,
      "c1_auc": 0.9400940623162846,
      "c2_sens99": 0.43333333333333335,
      "c2_auc": 0.9278953556731334,
      "baseline_ovr_sens99": 0.45,
      "baseline_ovr_auc": 0.9400940623162845
    }
  }
}
```

### D_engineered_features

```json
{
  "ov_auc": 0.9527784402575721,
  "ov_sens99": 0.25,
  "paad_auc": 0.9396237507348618,
  "paad_sens99": 0.45,
  "binary_sens99": 0.7355371900826446,
  "n_features": 63266
}
```

### E_class_weight_5x

```json
{
  "ov_auc": 0.9520033388981636,
  "ov_sens99": 0.25,
  "paad_auc": 0.9391534391534392,
  "paad_sens99": 0.4666666666666667,
  "binary_sens99": 0.7355371900826446,
  "weight": 5.0
}
```

### E_class_weight_10x

```json
{
  "ov_auc": 0.953076556165037,
  "ov_sens99": 0.25,
  "paad_auc": 0.9407701352145796,
  "paad_sens99": 0.4666666666666667,
  "binary_sens99": 0.7355371900826446,
  "weight": 10.0
}
```

### F_margin_features

```json
{
  "ov_auc": 0.9545671357023612,
  "ov_sens99": 0.25,
  "paad_auc": 0.9398001175778954,
  "paad_sens99": 0.45,
  "binary_sens99": 0.7355371900826446
}
```

### G_B_avg_E5

```json
{
  "ov_auc": 0.9580252802289531,
  "ov_sens99": 0.35714285714285715,
  "paad_auc": 0.9346413874191652,
  "paad_sens99": 0.4166666666666667,
  "binary_sens99": 0.7355371900826446,
  "members": "B_avg_E5"
}
```

### G_B_avg_E10

```json
{
  "ov_auc": 0.9584724540901502,
  "ov_sens99": 0.35714285714285715,
  "paad_auc": 0.9355526161081716,
  "paad_sens99": 0.4166666666666667,
  "binary_sens99": 0.7355371900826446,
  "members": "B_avg_E10"
}
```

### G_B_avg_base

```json
{
  "ov_auc": 0.9587109468161221,
  "ov_sens99": 0.35714285714285715,
  "paad_auc": 0.9352733686067018,
  "paad_sens99": 0.4,
  "binary_sens99": 0.7355371900826446,
  "members": "B_avg_base"
}
```

### G_B_avg_E5_avg_E10

```json
{
  "ov_auc": 0.9585320772716432,
  "ov_sens99": 0.32142857142857145,
  "paad_auc": 0.9379335684891241,
  "paad_sens99": 0.4166666666666667,
  "binary_sens99": 0.7355371900826446,
  "members": "B_avg_E5_avg_E10"
}
```

### G_all_four_avg

```json
{
  "ov_auc": 0.9591879322680658,
  "ov_sens99": 0.2857142857142857,
  "paad_auc": 0.9394473838918284,
  "paad_sens99": 0.43333333333333335,
  "binary_sens99": 0.7355371900826446,
  "members": "all_four_avg"
}
```

## Score distribution summary (technique A)

| Class | n | mean pos | std pos | skew pos | kurt pos | op_thr@99% | pos above thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRCA | 53 | 0.4119 | 0.4139 | 0.406 | -1.602 | 0.4273 | 22 |
| CRC | 27 | 0.4736 | 0.4380 | 0.010 | -1.888 | 0.6586 | 14 |
| HCC_J | 89 | 0.9336 | 0.2116 | -3.623 | 11.948 | 0.5663 | 84 |
| HEALTHY | 264 | 0.9081 | 0.2171 | -2.862 | 7.476 | 0.9997 | 85 |
| LUAD | 79 | 0.5579 | 0.3932 | -0.176 | -1.650 | 0.6159 | 40 |
| OTHER_C | 27 | 0.2858 | 0.3507 | 0.971 | -0.516 | 0.2549 | 11 |
| OV | 28 | 0.1676 | 0.2501 | 1.696 | 2.086 | 0.3551 | 7 |
| PAAD | 60 | 0.4295 | 0.4090 | 0.317 | -1.645 | 0.3891 | 27 |
