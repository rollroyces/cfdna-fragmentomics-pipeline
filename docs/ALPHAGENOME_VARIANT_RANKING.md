# AlphaGenome Variant Impact (AVI) ranking — feasibility report

**Date:** 2026-09-17
**Outcome:** **Not executable on this codebase** — scope mismatch.
**No model was trained on or modified using AlphaGenome Atlas outputs.**

---

## TL;DR

The task asked to "re-rank the per-cancer panel of mutations by AVI score
before training the LR classifier." The `cfdna-fragmentomics-pipeline`
has **no per-mutation panel**. The LR classifier is a tumor-naive
fragmentomics detector trained on 5-channel coverage features
(DELFI 5 Mb ratio + coverage + 100 kb ratio + counts + FSD-196 = 63 246
features per sample). There is no patient-specific mutation panel,
no variant calls, and no AVI-score input.

The hypothesis "variants with higher AVI score carry more cancer signal
per mutation than uniformly-tracked mutations" is testable in
`/Users/hermes/deepcatch/src/foundation/model.py` (DeepCatch's panel-LLR
detector that consumes TCGA MAFs), **not** in this pipeline.

**Did AVI-ranked panel beat uniform panel?** N/A — no panel exists to rank.
**Was the "no ML training" restriction honored?** Yes — no model was
trained or modified using AlphaGenome Atlas outputs.

---

## Evidence the pipeline has no per-mutation panel

| Source | Quote / finding |
|---|---|
| `USAGE.md:489` | "HEAD/tumor-naive (this project): the fragmentomics-only classifier that **does not require patient-specific mutation panels**. Contrast with the 'mutation-informed' panel-LLR detector in DeepCatch." |
| `scripts/honest_benchmark.py:55-65` (`load5`) | Loads only 5 `.npy`/`.fsd.json` files per sample: `delfi_5mb_ratio`, `delfi_5mb_coverage`, `delfi_100kb_ratio`, `delfi_100kb_counts`, `fsd`. No `.vcf`, no per-variant data. |
| `data/features/` directory listing | Contains only `.npy`, `.fsd.json`, `.delfi.json`, `.tsv`, and a few `.md` files. **No `.vcf`, `.maf`, `variants.*`, `mutations.*`, or `avi.*`** anywhere. |
| `labels_multiclass.tsv`, `labels_cross_study.tsv` | Sample-level disease labels only. No per-mutation rows. |
| `scripts/fusion_ablation.py:1-9` | "The synthetic mutation score is *not* a real mutation model — it's a calibrated stand-in." The 'mutation channel' in the existing fusion experiments is explicitly synthetic. |
| `docs/SENS_OPTIMIZATION.md` | The `frag+fusion 0.9921` figure uses a synthetic mutation stand-in, not AVI-derived variants. |

## Evidence the LR was trained on fragment coverage, not variants

| Source | Quote / finding |
|---|---|
| `scripts/multiclass_classification.py:42-46` | Imports `load5` and `fsd_vec` from `honest_benchmark`; `X.shape` is `(545, 63246)`. |
| `results/multiclass_classification.json` | `"n_samples": 545, "n_features": 63246`. 63 246 features = sum of DELFI windows + FSD-196. |
| `scripts/per_cancer_ov_paad_sweep.py:717-720` | "X.shape[0]} samples × {X.shape[1]} features". No mutation column. |
| `scripts/per_cancer_ov_paad_sweep.py:_ovr_oof` | The OvR loop fits `LogisticRegression` on `Xtr_p` (PCA-projected coverage features). There is no mutation column. |

## Evidence OV is borderline-excluded

| Source | Finding |
|---|---|
| `scripts/multiclass_classification.py:54` | `MIN_CLASS_N = 30` |
| `results/multiclass_classification.json` (`dropped_classes`) | `["CRC", "OTHER_C", "OV"]` are dropped. OV (n=28) only appears in the per-cancer sweep because that script sets `keep=set(labels.values())`. |
| `results/per_cancer_ov_paad.json` | The OV/PAAD sweep's cohort includes OV (n=28) but uses a non-default cohort; the binary `Sens@99% = 0.7355` is below the 0.755 floor stated in the task. The published frag-only reference (`results/sens_optimization_calibration.json`) is on a different cohort. |

---

## Why the AVI re-ranking hypothesis doesn't apply here

The AVI score is a per-variant scalar: AVI(`chr`, `pos`, `ref`, `alt`).
To "re-rank a per-cancer panel of mutations," the panel must already
contain variants. The cfdna-fragmentomics-pipeline's `X` has neither
variants nor a per-cancer panel column. Mapping the existing 63 246
features (windowed coverage) onto AVI scores is not a 1-to-1 transform:

1. A 5 Mb window covers ~5 000 000 bp and contains thousands of SNVs.
   AVI is per-SNV; the window is per-binned-coverage-summary. There is
   no principled way to "rank the panel by AVI" when the panel is a
   vector of 5 Mb coverage ratios.
2. Even if we constructed a per-cancer variant panel from external
   sources (TCGA MAFs / COSMIC), the AVI score itself cannot be a model
   input per the DeepMind terms. So the AVI score would have to
   influence a feature pre-LR (e.g. window weight), but that weight
   *is* a model input by the time it enters `X` — same restriction.
3. The 5-channel features are not patient-specific. Two healthy samples
   share the same 5-channel feature vector structure; there is no
   per-sample variant axis on which AVI could differentiate anything.

The honest bottom line: AVI ranking is **out of scope** for this
fragmentomics-only pipeline. It belongs in DeepCatch's panel-LLR
detector (which already consumes TCGA-observed variants and would be a
natural place to swap uniform panel weighting for AVI-ranked panel
weighting).

---

## What was actually done

| Step | Outcome |
|---|---|
| Read `per_cancer_ov_paad_sweep.py` and `per_cancer_auc.py` | Confirmed 5x5 OvR LR + PCA(200) protocol on 63 246-dim fragment features; no mutation column. |
| Read `multiclass_classification.py` | Confirmed OvR LR pipeline; confirmed OV is dropped at `MIN_CLASS_N=30`. |
| Installed AlphaGenome | `pip install alphagenome` succeeded in the `deepcatch` venv (v1.2.x). `alphagenome.models.dna_client.DnaClient` and `alphagenome.atlas.atlas.AtlasClient` confirmed importable. Atlas API calls require a Google Cloud API key; **no calls were made**. |
| Searched repo for any per-mutation data | None found. |
| Compared baseline OV/PAAD Sens@99% (from prior sweep at HEAD `5d4f3a9`) | OV 0.2500, PAAD 0.4500 — matches task description. |
| Computed the binary regression guard | `binary_sens99 = 0.7355` — already **below** the 0.755 floor. The floor is unmet in this cohort, independent of any new technique. |
| Ran the AVI-ranking experiment | **Not executed** — no panel to rank. |

---

## Results

Saved to `results/alphagenome_variant_ranking.json`.

```
ov_sens99_improvement:  0.0  (delta vs uniform panel — not measurable)
paad_sens99_improvement: 0.0  (delta vs uniform panel — not measurable)
ov_auc_floor_preserved:   true  (0.9546 > 0.94, baseline)
paad_auc_floor_preserved: true  (0.9401 > 0.94, baseline)
```

The baseline per-cancer AUCs are unchanged because the pipeline was not
retrained. The Sens@99% deltas are zero because there is no experiment
to measure.

---

## Recommendations

1. **Do not pursue AVI ranking in this pipeline.** It is the wrong
   architecture for the hypothesis. The feature space is windowed
   coverage, not variants.
2. **Pursue AVI ranking in `/Users/hermes/deepcatch/src/foundation/`.**
   That panel-LLR detector consumes real TCGA-observed mutations and
   would let us honestly test "AVI-ranked panel > uniform panel" by
   re-ranking the per-patient variant list at score time (AVI score as
   panel-weight, not as model input — preserving the DeepMind terms).
3. **Address the underlying OV/PAAD Sens@99% problem differently.** The
   prior `per_cancer_ov_paad_sweep` tried 13 feature-engineering
   techniques and none hit the targets (best OV 0.357, best PAAD 0.467,
   both below the 0.40 / 0.60 targets). The remaining lever is
   data-quality (OV n=28 is at the threshold of statistical
   detectability for any high-specificity cut) or moving to a
   fragment-length-aware channel (mean length at 100 kb is already in
   the 8-channel set; per-cancer 8-channel evaluation might reveal
   length signals invisible to ratio+coverage alone).
4. **Reconcile the binary regression floor.** The published frag-only
   reference AUC is 0.9755 on a cohort that drops OV. The OV/PAAD sweep
   uses a different cohort that includes OV. The 0.755 binary-Sens@99%
   floor cannot be evaluated against two different cohorts in the same
   statement.

---

## Honor check (DeepMind Terms)

- [x] **No model trained on AlphaGenome Atlas outputs.** No model was
      trained, refit, or modified in this work. The AVI score never
      appeared in any feature matrix, scaler, PCA component, or LR
      coefficient.
- [x] **AVI used only for rank / selection** — N/A, because there was
      nothing to rank. The "rank-then-train" pattern was never
      instantiated.
- [x] **No Atlas API calls made.** The API client was imported to verify
      availability; no variants were queried.
- [x] **No synthetic / invented data.** The `results/alphagenome_variant_ranking.json`
      reports `0.0` deltas with explicit "scope mismatch" framing
      rather than fabricating a delta.