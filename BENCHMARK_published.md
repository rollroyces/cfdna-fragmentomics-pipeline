# Published-test benchmark comparison

Side-by-side sensitivity-at-specificity comparison of multi-cancer
early-detection (MCED) tests at their reported headline operating points.
This is **Enhancement #2** of the fragmentomics decision-curve work.

## Headline comparison

| Test                                  | Cohort | Cancer types | Sens @ Spec                        | Source                                         |
|---------------------------------------|-------:|-------------:|------------------------------------|------------------------------------------------|
| **Galleri** (CCGA-3, sub-study 3)     |  4,023 |         50+  | **51.5%** @ 99.5% (prespecified)   | Klein et al., *Ann Oncol* 2021; CCGA-3 fact sheet 2024 (specificity 99.5%, 95% CI 99.0–99.8) |
| **Shield** (Guardant, ECLIPSE)        |  7,861 |     CRC only | **83.1%** @ 89.6%                  | Chung et al., *NEJM* 2024 / ECLIPSE; Lancet Gastroenterology 2024 editorial (n=7,861 average-risk, colonoscopy reference) |
| **CancerSEEK** (multi-cancer)         |  1,005 |            8 | **median 70%** @ >99% (range 69–98% by type) | Cohen et al., *Science* 2018 (1,005 non-metastatic cancers; 7/812 healthy false positives → spec ~99.1%) |
| **This — fragmentomics (cross-study)**| **627**|            8 | **75.5%** @ 99%  (95% CI 63.3–87.6) | `honest_benchmark.py` + `scripts/sens_at_specificity.py`, pooled 5-seed 5-fold OOF, 5-channel PCA n=200 + per-study harmonize |
| **This — fragmentomics + simulated mutation fusion** | **627** | 8 | **84.3%** @ 99% (95% CI 80.0–92.9) | `scripts/fusion_ablation.py`, naive-avg with synthetic-mutation channel (AUC 0.95, calibrated to DeepCatch's panel-LLR @ 0.1% VAF) |

### Full per-specificity table (this work)

| Specificity | Sens (fragmentomics only) | 95% CI (bootstrap, n=1000) | Sens (fragmentomics + fusion) | 95% CI (bootstrap, n=1000) |
|------------:|--------------------------:|---------------------------:|------------------------------:|---------------------------:|
|        95%  | **89.8%**                 | 84.3% – 93.2%              | **93.9%**                     | 90.4% – 98.7%              |
|        98%  | **85.4%**                 | 73.4% – 89.7%              | **91.5%**                     | 83.2% – 94.7%              |
|        99%  | **75.5%**                 | 63.3% – 87.6%              | **84.3%**                     | 80.0% – 92.9%              |
|      99.5%  | **75.5%**                 | 61.6% – 86.7%              | **84.0%**                     | 78.9% – 92.2%              |
|      99.9%  | **64.7%**                 | 60.5% – 80.1%              | **81.3%**                     | 77.4% – 89.8%              |

**Cohort:** 627 samples (363 cancer + 264 healthy) pooled across the
Cristiano (CRC / breast / lung / ovarian / pancreatic / gastric) and
Jiang (HCC) FinaleDB studies. 5-channel feature set: 5Mb-arm ratio +
5Mb-arm coverage + 100kb-bin ratio + 100kb-bin counts + 196-bin
fragment-size distribution.

**AUC (pooled OOF, 5-seed mean):** 0.9755 ± 0.0016 (fragmentomics only);
0.9921 (naive-avg fusion). Per-seed std < 0.005 — well within stochastic
LR-convergence noise (~1-3pp run-to-run on the central sens estimate;
the bootstrap CIs already absorb this).

## Honest caveats (READ BEFORE QUOTING)

1. **Cohort size.** n = 627 vs Galleri's n = 4,023, Shield's n = 7,861,
   CancerSEEK's n = 1,005. **All four published numbers come from
   prospectively-enrolled clinical cohorts, not pooled OOF.**
   A 6–10× cohort-size gap means our CIs are wider, and a positive
   result is more likely to be fragile on external validation.

2. **No external validation.** The 75.5% / 84.3% Sens@99% numbers
   above are **pooled out-of-fold** predictions on the same 627
   samples the model was trained on (5-seed × 5-fold CV, with
   per-fold harmonization). This is the standard ML benchmark
   protocol, but **it is not the same as held-out external
   validation**. A test that achieves 85% Sens@99% on pooled OOF
   routinely drops to 70–80% on a fresh cohort (see e.g. Galleri's
   PATHFINDER follow-up).

3. **Cancer-type mix is different.** Galleri's headline 51.5%
   includes 50+ cancer types with a stage-shift toward late-stage;
   this work pools 8 cancer types with the Cristiano-HCC cohort,
   which skews toward easier-detected types (CRC, breast, HCC).
   A 1:1 apples-to-apples comparison would re-run Galleri /
   CancerSEEK on this 627-sample cohort — not done here.

4. **The mutation-channel fusion is synthetic.** The "fragmentomics
   + fusion" row averages the pooled OOF fragmentomics score with
   a *calibrated synthetic* mutation score tuned to DeepCatch's
   panel-LLR headline (AUC 0.92, Sens@95% 0.77). It is intended
   as an upper-bound ablation: a real mutation-informed fusion
   requires the deepcatch mutation model run on the same 627
   samples, which is **not** done in this repo. For the real
   mutation-fusion ablation see
   `deepcatch/src/fragmentomics/fusion_ablation.py`.

5. **Bootstrap CIs reflect within-cohort uncertainty only.**
   The bootstrap resamples the 627 samples (with replacement) and
   recomputes Sens@Spec on each resample. It does **not** capture
   external-cohort generalization uncertainty. A realistic
   generalization CI would be substantially wider (often ±15-20pp
   at high specificity).

6. **Operating-point selection.** The published tests report Sens
   at their *own* prespecified specificity (Galleri 99.5%, Shield
   89.6%, CancerSEEK ~99.1%). To compare like-with-like we report
   Sens at 99% here — the most common MCED screening operating
   point. A 0.5pp–10pp spec shift can change Sens by 5–15pp at
   these specificities, so the headline numbers are not directly
   subtractable. In particular, Shield's lower spec (89.6%) is a
   major reason its sens (83.1%) is much higher than Galleri's
   (51.5%): comparing them at equal spec would shrink the gap.

## What changed compared to the previous decision-curve

- **`scripts/sens_at_specificity.py`** (new): pooled 5-seed OOF +
  per-spec report with bootstrap CI. Writes `results/sens_at_spec.json`.
- **`scripts/fusion_ablation.py`** (new): naive-avg fusion with a
  synthetic mutation channel. Writes `results/fusion_ablation.json`.
- **`scripts/ppv_screening.py`** (extended): new `--sens-json` flag
  loads the sens@spec table and appends PPV-at-each-prevalence rows
  + a printed sens@spec summary. No removal of legacy operating points.
- **`test/test_sens_at_specificity.py`** (new, 4 tests):
  - sens@95% is monotonically non-increasing as spec rises
  - bootstrap CIs are non-negative
  - sens@95% matches honest_benchmark's reported value within ±3σ
  - the BENCHMARK_published.md row fields are reproducible from JSON

## Reproducing

```bash
# 1. Compute pooled OOF + sens@spec with CIs
python scripts/sens_at_specificity.py --save-scores \
    --out results/sens_at_spec.json --n-bootstrap 1000

# 2. Compute fusion ablation (loads sens_at_spec_scores.npz)
python scripts/fusion_ablation.py \
    --sens-json results/sens_at_spec.json \
    --out results/fusion_ablation.json

# 3. Re-emit PPV/screening numbers with the new sens@spec operating
#    points (writes results/ppv_screening.json)
python scripts/ppv_screening.py \
    --sens-json results/sens_at_spec.json \
    --out results/ppv_screening.json

# 4. Verify
pytest test/test_sens_at_specificity.py -v
ruff check scripts/sens_at_specificity.py scripts/fusion_ablation.py \
        scripts/ppv_screening.py test/test_sens_at_specificity.py
```
