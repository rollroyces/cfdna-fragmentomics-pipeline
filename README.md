# cfdna-fragmentomics-pipeline

**Tumor-naive cell-free DNA (cfDNA) fragmentomics pipeline — from raw sequencing to cancer classification.**

A production-oriented pipeline that extracts the "Big Three" fragmentomic signatures from cfDNA and classifies Cancer vs Healthy **without any prior knowledge of tumor mutations**. Built for reproducibility (Snakemake), honest validation (fixed-specificity metrics, no threshold optimization), and real data (FinaleDB public cohort, no synthetic data).

> **How it compares to industry:** see [BENCHMARK.md](BENCHMARK.md) — at parity with DELFI-HCC on the clinically relevant metric (86% vs 84.5% sensitivity @ 95% specificity), with an honest accounting of where the comparison is and isn't fair.

---

## Installation

```bash
# Pure-Python install (no Java, no Snakemake required for the scripts)
pip install -e .

# With Snakemake workflow execution
pip install -e ".[snakemake]"

# With BAM-mode pysam support
pip install -e ".[bam]"

# After install, four CLI entry points are globally available:
cfdna-fetch --help        # fetch FinaleDB samples
cfdna-classify --help     # train classifier + 5-seed CV
cfdna-fsd --help          # fragment-size extraction
cfdna-delfi --help        # DELFI ratio + coverage extraction

# For the multi-cohort honest benchmark, run directly from the repo:
python scripts/honest_benchmark.py --help  # 10+ minute multi-cohort run
```

## The Big Three fragmentomic features

| Feature | Biology | Signal |
|---|---|---|
| **Fragment Size Distribution (FSD)** | cfDNA wraps around nucleosomes. Healthy plasma peaks at ~167 bp (nucleosome + linker); tumor-derived cfDNA is shorter, ~145 bp | Median/mode shift toward short fragments, elevated short(100-150)/long(150-220) ratio |
| **4-mer End-Motif Frequencies** | Different nucleases in the tumor microenvironment (e.g. DNASE1L3) cut cfDNA at different 5' ends | Frequency shift of the 256 possible 4-mer motifs (e.g. CCCA) |
| **DELFI / Window Protection Score (WPS)** | Regional fragmentation reflects chromatin accessibility & copy number: open chromatin fragments differently than silent regions | Per-100kb-bin short/long ratio (DELFI); WPS = fragment full-coverage minus boundary events |

## Architecture

```
cfdna-fragmentomics-pipeline/
├── data/
│   ├── raw/                 # .frag.tsv.bgz or .bam (git-ignored)
│   └── references/          # hg38 per-bin GC reference
├── envs/environment.yml     # conda env: numpy, scipy, sklearn, pysam, snakemake
├── modules/                 # (optional) per-rule Snakemake modules
├── scripts/
│   ├── fetch_finaledb.py    # FinaleDB API → real cfDNA fragment records (streaming)
│   ├── extract_fsd.py       # FSD from frag.tsv (FinaleDB) or BAM TLEN (pysam)
│   ├── extract_delfi.py     # 100kb/5Mb bins: short/long ratio + WPS
│   ├── extract_motifs.py    # 4-mer end motifs (BAM mode, needs reference FASTA)
│   ├── build_gc_reference.py# hg38 per-100kb GC fractions
│   ├── gc_correction.py     # LOESS regression correction for GC bias
│   └── train_classifier.py  # RF/GB, stratified CV, fixed-specificity metrics
├── main.smk                 # Snakemake master workflow
├── config.yaml
├── run_real_cohort.py       # single-machine streaming path (laptop-friendly)
└── README.md
```

## Two execution modes

### Mode A — FinaleDB pre-processed data (no local BAM processing)

Fetches **real** cfDNA fragment records (2,500+ samples: Liver cancer, Colorectal cancer, Breast cancer, Healthy controls, ...) from the [FinaleDB](https://pubmed.ncbi.nlm.nih.gov/33258919/) public S3 bucket. Per-fragment rows are `chrom start end mapq strand`. No alignment needed — FinaleDB used a uniform pipeline (BWA-MEM + samblaster dedup + MAPQ30).

```bash
# one-shot cohort run (streams, disk-friendly)
python run_real_cohort.py --cancer "Liver cancer" --healthy \
    --n-cancer 8 --n-healthy 8 --out results

# or via Snakemake
snakemake -s main.smk --configfile config.yaml --cores 4
```

### Mode B — local BAMs (production path for labs)

Place deduplicated BAMs in `data/raw/<sample>.mdups.bam` and set `mode: "bam"` in `config.yaml`. Extractors use pysam with the strict contract: **properly-paired reads, MAPQ ≥ 30**. 4-mer motifs require a reference FASTA (`reference_fasta` in config).

## Real-data result (cross-study, 627 samples)

**AUC 0.9745 ± 0.0023 (5-seed CV, 95% CI ≈ ±0.005)** — 363 cancer vs 264
healthy cfDNA WGS samples across two low-pass studies (Jiang 2015 +
Cristiano 2019), per-study z-score harmonized. Logistic Regression on a
**5-channel fragmentomic profile** (5Mb + 100kb short/long ratio,
5Mb + 100kb median-normalized coverage, FSD size histogram),
PCA to 200 components inside each CV fold (5-fold, pooled out-of-fold,
no leakage).

(Numbers from `python scripts/honest_benchmark.py` on the 627 cohort;
single-seed re-runs vary by ±0.001 AUC.)

Reproduce: `python run_cross_study.py --parallel 8 --max-mb 500`

Single-study (Jiang 2015, 121 samples): **AUC 0.9716 ± 0.003, Sens@95% 0.894, Sens@99% 0.811**.

**Feature ablations** (3-seed pooled OOF CV, per-study harmonized):

| Feature set | AUC (3-seed) | Δ vs base |
|---|---|---|
| 5-channel baseline (5Mb + 100kb + FSD) | 0.875 ± 0.011 | — |
| + 4-mer end motifs (256 bins) | 0.880 ± 0.012 | +0.005 (sub-noise) |
| + per-bin mean fragment length | 0.869 ± 0.011 | −0.006 (redundant) |

**Result**: 4-mer motifs +0.005 (below n=98 noise floor of ±0.011);
mean-length redundant with the short/long ratio. The 5-channel profile is
near-optimal — no re-extraction of 627 samples justified for sub-noise
gain.

## Honest validation

- **Stratified K-fold CV** (LOOCV when n < 25) — never trains on test samples.
- **Fixed-specificity sensitivity**: sens@95% and sens@99% spec, read off the ROC at that specificity — **no threshold optimization on test predictions**.
- **No leakage**: PCA and per-study harmonization are fit on training folds only.
- **Data-quality guards**: deep-WGS file-size rejection, cell-line exclusion (GM1100 mislabeled "Liver cancer", GM\* cirrhosis/HBV lines), study filtering.
- **GC-bias correction**: LOESS (tricube-weighted local regression) fit of per-bin coverage vs GC fraction.
- **Real data only**: no synthetic cohorts. FinaleDB samples are real clinical cfDNA WGS.

## Expected runtimes

| Step | Per sample | Notes |
|---|---|---|
| Fetch frag.tsv.bgz (FinaleDB) | 1-3 min | ~170 MB, streamed |
| FSD extraction | ~5 s | 28M fragments |
| DELFI + WPS | ~5 s | 100 kb bins × 30,894 |
| GC correction | <1 s | LOESS, 30k bins |
| Motifs (BAM mode) | ~2 min | pysam + reference FASTA |
| Classify (RF, 5-fold) | <1 min | ~16 features |

## Reproducibility

- `envs/environment.yml` — conda environment (Python 3.11, numpy/scipy/sklearn/pysam/snakemake)
- All random seeds fixed in the classifier (`random_state=42`)
- FinaleDB data is versioned upstream (uniform processing pipeline)
- Full result JSON in `results/classifier_results.json`

## License

MIT
