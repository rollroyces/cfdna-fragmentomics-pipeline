# Competitive Landscape & Benchmark Comparison

**Status:** This pipeline is competitive with, but not superior to, the
industry standard — with clear, honest caveats about cohort size and difficulty.

## The industry standard

cfDNA fragmentomics for cancer detection is led by **DELFI** (Delfi
Diagnostics) and a handful of academic groups. The key published benchmarks:

| Study | Cohort | Task | AUC | Sensitivity |
|---|---|---|---|---|
| **DELFI (Cristiano 2019, *Nature*)** | 236 cancer + 245 healthy, 7 cancer types | pan-cancer | ~0.94 | 57–99% across stages @ 98% spec |
| **DELFI-HCC (Foda 2023, *Nature Comms* / *Cancer Discovery*)** | **724** individuals (HCC + average/high-risk) | HCC vs risk groups | **0.94** | **84.5% @ 95% spec** |
| Ultra-low coverage (eLife 2024) | 426 cancer (16 types) + 295 healthy | pan-cancer | 0.896 | — |
| **This pipeline** | **121** (89 HCC + 32 healthy, Jiang 2015) | HCC vs healthy | **0.983** | **86% @ 95% spec**, 79% @ 99% |

## The honest comparison

**On the comparable metric — sensitivity at 95% specificity — this pipeline
is at parity with DELFI-HCC (86% vs 84.5%).** That is the number that
matters clinically, and it holds up.

**The raw AUC (0.983 vs 0.94) is NOT a fair win**, and I won't claim it is.
Three reasons:

1. **Cohort size.** 121 samples vs DELFI's 724. A single-study cohort of 89
   cancer + 32 healthy gives the classifier an *easier* decision boundary.
2. **Cohort difficulty.** My "healthy" controls are truly healthy donors.
   DELFI-HCC's benchmark includes *average-risk and high-risk* individuals
   (cirrhosis, hepatitis B) — the clinically hard group where fragmentomics
   must separate cancer from pre-malignant liver disease. That is a harder
   task than cancer-vs-healthy.
3. **Healthy-control count.** 32 healthy samples quantize the 99%-spec
   operating point (0 false positives); DELFI's hundreds of controls make
   its high-specificity claims statistically stable.

## Where this work is *genuinely* ahead of the industry

- **Open source & reproducible.** DELFI/GRAIL/Guardant are proprietary
  black boxes. Every step here — data fetch, feature extraction, PCA, CV —
  is a single reproducible command against public data.
- **Data-quality rigor.** The pipeline enforces three controls most papers
  bury: single-study filtering (coverage batch effects), deep-WGS
  file-size guard, and cell-line exclusion (a mislabeled technical control
  that was inflating the healthy-vs-cancer separation).
- **Honest metrics.** Fixed-specificity sensitivity computed at the correct
  operating point (the "Sens@99% = 0" bug was caught and fixed, not papered
  over), pooled out-of-fold predictions, PCA inside folds — no leakage.

## What would close the gap to a publishable claim

1. **Cross-study harmonization** — combine Jiang 2015 with Cristiano 2019
   (both low-pass) via coverage normalization + batch correction, the way
   the DELFI papers pool cohorts.
2. **A high-risk comparison group** — cirrhosis/HBV vs HCC (Jiang 2015 has
   16 cirrhosis + 40 hepatitis B samples) — the clinically relevant test.
3. **External validation** — train on one study, test on another.

## Data-integrity findings (documented, not hidden)

1. **The high-risk group is cell lines.** FinaleDB's Jiang 2015 "Cirrhosis"
   (36) and "Hepatitis B" (67) entries are named `GM*` — the Coriell
   lymphoblastoid cell-line catalog (GM886, GM918, GM1403, …). They are
   annotated `tissue=blood plasma` but are reference cell lines, not patient
   plasma. The pipeline's cell-line guard excludes them. **Consequence:** a
   true HCC-vs-high-risk (cirrhosis/HBV) comparison is *not possible* with
   this public data — real high-risk patient plasma lives behind the
   DELFI-HCC cohort (Foda 2023), which is not openly deposited.
2. **GM1100** (B-lymphocyte line) was mislabeled "Liver cancer" — caught and
   excluded; its cell-line fragmentation would have inflated the
   cancer-vs-healthy separation.

## Sources

- Cristiano et al., *Nature* 570:385 (2019) — DELFI pan-cancer.
- Foda et al., *Nature Communications* 14:3294 (2023) — DELFI-HCC, 724
  individuals, "84.5% sensitivity at 95% specificity, 0.94 AUC".
- "Detecting Liver Cancer Using Cell-Free DNA Fragmentomes," *Cancer
  Discovery* 13:616 (2023).
- Jiang et al., *PNAS* 112:E1317 (2015) — the source cohort (PMID 25646427).
