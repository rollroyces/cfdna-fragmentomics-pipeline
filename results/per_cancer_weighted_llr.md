# Per-cancer CADD-style continuous channel weighting

Generated: 2026-09-17T10:20:22.886490+00:00
Cohort: 627 samples x 63246 features

**Schemes evaluated:** linear, sigmoid_10. Primary (used for integration decisions): `linear`.

**Weight application:** per-channel column-wise scaling (`X_weighted[:, j] = X[:, j] * w_j`) before standardization + PCA(200) + OvR LR. Weights derived from inner 3-fold CV per-channel Mann-Whitney U, computed on the training fold only (no test leakage).

**Inner-CV ranking:** 3-fold, seed=42  |  **Final eval:** 5 seeds x 5 folds pooled OOF

## Per-cancer results (weighted LR vs uniform-baseline vs top-K)

### Scheme: `linear`

| Cancer | Sens@99% (weighted) | Sens@99% (uniform base) | Sens@99% (top-K) | AUC (weighted) | AUC (uniform base) | AUC (top-K) | Sens delta vs base | Sens delta vs topK | Target met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OV | 0.0714 | 0.2500 | 0.3571 | 0.9438 | 0.9546 | 0.9525 | -0.1786 | -0.2857 | FAIL (0.40) |
| PAAD | 0.3833 | 0.4500 | 0.4333 | 0.9391 | 0.9401 | 0.9449 | -0.0667 | -0.0500 | FAIL (0.55) |
| BRCA | 0.3396 | 0.4151 | 0.3396 | 0.9543 | 0.9532 | 0.9540 | -0.0755 | +0.0000 | FAIL (0.50) |

### Scheme: `sigmoid_10`

| Cancer | Sens@99% (weighted) | Sens@99% (uniform base) | Sens@99% (top-K) | AUC (weighted) | AUC (uniform base) | AUC (top-K) | Sens delta vs base | Sens delta vs topK | Target met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OV | 0.2500 | 0.2500 | 0.3571 | 0.9530 | 0.9546 | 0.9525 | +0.0000 | -0.1071 | FAIL (0.40) |
| PAAD | 0.4500 | 0.4500 | 0.4333 | 0.9400 | 0.9401 | 0.9449 | +0.0000 | +0.0167 | FAIL (0.55) |
| BRCA | 0.4151 | 0.4151 | 0.3396 | 0.9553 | 0.9532 | 0.9540 | +0.0000 | +0.0755 | FAIL (0.50) |

## Binary regression guard (cancer vs healthy, full features)

* Binary Sens@99% (pooled, 5x5 OOF): **0.7769**  (uniform baseline 0.7355, topK 0.7768595041322314, floor 0.755) ✅ baseline preserved, ✅ floor met
* Binary AUC (pooled, 5x5 OOF): **0.9698**  (uniform baseline 0.9660, topK 0.9698013189748727, floor 0.9755) ✅ baseline preserved, ❌ floor met

## Per-cancer weight statistics (across all outer folds)

| Scheme | Cancer | n_active_mean | n_active_min | n_active_max | mean_w_active | max_w_mean | median_w |
|---|---|---:|---:|---:|---:|---:|---:|
| linear | OV | 47651 | 46821 | 48441 | 0.1258 | 0.3448 | 0.1212 |
| linear | PAAD | 49619 | 49213 | 49905 | 0.1158 | 0.3542 | 0.0844 |
| linear | BRCA | 44609 | 42802 | 45482 | 0.0916 | 0.3429 | 0.0810 |
| sigmoid_10 | OV | 63246 | 63246 | 63246 | 0.6595 | 0.9689 | 0.7689 |
| sigmoid_10 | PAAD | 63246 | 63246 | 63246 | 0.6347 | 0.9718 | 0.6989 |
| sigmoid_10 | BRCA | 63246 | 63246 | 63246 | 0.5748 | 0.9685 | 0.6912 |

## Per-seed stability (5-seed x 5-fold pooled OOF, continuous weighting)

| Scheme | Cancer | AUC mean | AUC std | AUC min-max | Sens@99% mean | Sens@99% std | Sens@99% min-max |
|---|---|---:|---:|---:|---:|---:|---:|
| linear | OV | 0.9381 | 0.0081 | 0.9276..0.9518 | 0.0571 | 0.0484 | 0.0000..0.1429 |
| linear | PAAD | 0.9357 | 0.0059 | 0.9307..0.9461 | 0.3767 | 0.0655 | 0.2833..0.4667 |
| linear | BRCA | 0.9519 | 0.0067 | 0.9411..0.9597 | 0.3132 | 0.0350 | 0.2642..0.3585 |
| sigmoid_10 | OV | 0.9519 | 0.0048 | 0.9433..0.9569 | 0.1786 | 0.0505 | 0.1071..0.2500 |
| sigmoid_10 | PAAD | 0.9394 | 0.0058 | 0.9319..0.9494 | 0.4067 | 0.0429 | 0.3667..0.4667 |
| sigmoid_10 | BRCA | 0.9497 | 0.0109 | 0.9297..0.9622 | 0.4000 | 0.0669 | 0.2830..0.4717 |

## Bottom line

Per-cancer CADD-style continuous weighting (this script) vs uniform-channel OvR baseline vs hard top-K subset (commit b5d8260):

  Scheme: linear
    OV: Sens@99% 0.2500 -> 0.0714 (-0.1786 vs uniform; -0.2857 vs top-K), AUC 0.9546 -> 0.9438 (-0.0107 vs uniform), target 0.40 -> NOT MET
    PAAD: Sens@99% 0.4500 -> 0.3833 (-0.0667 vs uniform; -0.0500 vs top-K), AUC 0.9401 -> 0.9391 (-0.0010 vs uniform), target 0.55 -> NOT MET
    BRCA: Sens@99% 0.4151 -> 0.3396 (-0.0755 vs uniform; +0.0000 vs top-K), AUC 0.9532 -> 0.9543 (+0.0011 vs uniform), target 0.50 -> NOT MET

  Scheme: sigmoid_10
    OV: Sens@99% 0.2500 -> 0.2500 (+0.0000 vs uniform; -0.1071 vs top-K), AUC 0.9546 -> 0.9530 (-0.0016 vs uniform), target 0.40 -> NOT MET
    PAAD: Sens@99% 0.4500 -> 0.4500 (+0.0000 vs uniform; +0.0167 vs top-K), AUC 0.9401 -> 0.9400 (-0.0001 vs uniform), target 0.55 -> NOT MET
    BRCA: Sens@99% 0.4151 -> 0.4151 (+0.0000 vs uniform; +0.0755 vs top-K), AUC 0.9532 -> 0.9553 (+0.0021 vs uniform), target 0.50 -> NOT MET

Binary guard (cancer vs healthy, full 63246 features):
  Sens@99%=0.7769  uniform-baseline=0.7355 (+0.0413, ABOVE baseline), floor=0.755 (MET)
  AUC     =0.9698  uniform-baseline=0.9660 (+0.0038, ABOVE baseline), floor=0.9755 (NOT MET)

Linear (literal task spec) beats hard top-K on: ['BRCA']
Linear regressed >1pp vs top-K on: ['OV', 'PAAD']
Linear stable across seeds (Sens@99% std < 0.10): ['OV', 'PAAD', 'BRCA']

No cancer gained >= 0.10 absolute Sens@99% over the uniform baseline while preserving AUC and binary baseline. Per-cancer picture (Sens@99% delta vs uniform, linear scheme): OV -0.1786, PAAD -0.0667, BRCA -0.0755.

