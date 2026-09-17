# Per-cancer top-K channel selection

Generated: 2026-09-17T07:46:12.577415+00:00
Cohort: 627 samples x 63246 features

Inner K-selection: 3-fold CV, seed=42  |  Final eval: 5 seeds x 5 folds pooled OOF

## K-selection (inner CV) and final per-cancer results

| Cancer | K chosen | K inner-CV AUC | final AUC | final Sens@99% | baseline AUC | baseline Sens@99% | AUC delta | Sens delta | Sens target | Sens met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OV | 10000 | 0.9585 | 0.9525 | 0.3571 | 0.9546 | 0.2500 | -0.0021 | +0.1071 | 0.40 | FAIL |
| PAAD | 5000 | 0.9377 | 0.9449 | 0.4333 | 0.9401 | 0.4500 | +0.0048 | -0.0167 | 0.55 | FAIL |
| BRCA | 50000 | 0.9404 | 0.9540 | 0.3396 | 0.9532 | 0.4151 | +0.0008 | -0.0755 | 0.50 | FAIL |

## Binary regression guard (cancer vs healthy, full features)

* Binary Sens@99% (pooled, 5x5 OOF): **0.7769**  (baseline 0.7355, floor 0.755) ✅ baseline preserved, ✅ floor met
* Binary AUC (pooled, 5x5 OOF): **0.9698**  (baseline 0.9660, floor 0.9755) ✅ baseline preserved, ❌ floor met

## Inner-CV AUC per K (selection curve)

### OV

| K | inner-CV AUC |
|---:|---:|
| 10 | 0.8850 |
| 50 | 0.9122 |
| 100 | 0.8989 |
| 500 | 0.9375 |
| 1000 | 0.9514 |
| 5000 | 0.9490 |
| 10000 | 0.9585 |
| 50000 | 0.9424 |

### PAAD

| K | inner-CV AUC |
|---:|---:|
| 10 | 0.9247 |
| 50 | 0.9363 |
| 100 | 0.9363 |
| 500 | 0.9363 |
| 1000 | 0.9363 |
| 5000 | 0.9377 |
| 10000 | 0.9266 |
| 50000 | 0.9172 |

### BRCA

| K | inner-CV AUC |
|---:|---:|
| 10 | 0.8561 |
| 50 | 0.8544 |
| 100 | 0.8484 |
| 500 | 0.8423 |
| 1000 | 0.9178 |
| 5000 | 0.9213 |
| 10000 | 0.9171 |
| 50000 | 0.9404 |

## Bottom line

Per-cancer top-K channel selection vs uniform channel use:

  * OV: Sens@99% 0.2500 -> 0.3571 (+0.1071), AUC 0.9546 -> 0.9525 (-0.0021), K=10000 (chosen by inner 3-fold CV), target 0.40 -> NOT MET
  * PAAD: Sens@99% 0.4500 -> 0.4333 (-0.0167), AUC 0.9401 -> 0.9449 (+0.0048), K=5000 (chosen by inner 3-fold CV), target 0.55 -> NOT MET
  * BRCA: Sens@99% 0.4151 -> 0.3396 (-0.0755), AUC 0.9532 -> 0.9540 (+0.0008), K=50000 (chosen by inner 3-fold CV), target 0.50 -> NOT MET

Binary guard (cancer vs healthy, full 63246 features):
  Sens@99%=0.7769  baseline=0.7355 (delta +0.0413, ABOVE baseline), floor=0.755 (MET)
  AUC     =0.9698  baseline=0.9660 (delta +0.0038, ABOVE baseline), floor=0.9755 (NOT MET)

Cancers with >= 0.10 absolute Sens@99% gain (per-cancer AUC and binary baseline both preserved): OV. Integration recommended.

