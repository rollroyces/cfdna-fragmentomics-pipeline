# Per-cancer AUC table (OvR, pooled 5-seed × 5-fold OOF)

Baseline protocol: LR (C=1.0) + PCA(200) on the 5-channel feature
set, harmonized per study. Same as `multiclass_classification.py`.

| Cancer | n_pos | AUC (pooled) | AUC 95% CI | Sens@99% (pooled) | Sens@99% 95% CI | Notes |
|---|---:|---:|---|---:|---|---|
| BRCA | 53 | 0.9532 | [0.9302, 0.9729] | 0.4151 | [0.2553, 0.6042] |  |
| CRC | 27 | 0.9743 | [0.9503, 0.9904] | 0.5185 | [0.3214, 0.7667] | n<30 (exploratory) |
| HCC_J | 89 | 0.9969 | [0.9936, 0.9993] | 0.9438 | [0.8823, 0.9880] |  |
| HEALTHY | 264 | 0.9698 | [0.9576, 0.9808] | 0.3220 | [0.1423, 0.6667] |  |
| LUAD | 79 | 0.9607 | [0.9355, 0.9805] | 0.5063 | [0.2169, 0.7000] |  |
| OTHER_C | 27 | 0.9541 | [0.9162, 0.9820] | 0.4074 | [0.1923, 0.6923] | n<30 (exploratory) |
| OV | 28 | 0.9546 | [0.9297, 0.9744] | 0.2500 | [0.0333, 0.4484] | n<30 (exploratory) |
| PAAD | 60 | 0.9401 | [0.9054, 0.9697] | 0.4500 | [0.1914, 0.6087] |  |
