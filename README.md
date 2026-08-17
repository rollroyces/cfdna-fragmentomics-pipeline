# cfdna-fragmentomics-pipeline

**Tumor-naive cell-free DNA (cfDNA) fragmentomics pipeline — from raw sequencing to cancer classification.**

A production-oriented pipeline that extracts the "Big Three" fragmentomic signatures from cfDNA and classifies Cancer vs Healthy **without any prior knowledge of tumor mutations**. Built for reproducibility (Snakemake), honest validation (fixed-specificity metrics, no threshold optimization), and real data (FinaleDB public cohort, no synthetic data).

---

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

## Honest validation

- **Stratified K-fold CV** (LOOCV when n < 25) — never trains on test samples.
- **Fixed-specificity sensitivity**: sens@95% and sens@99% spec, read off the ROC at that specificity — **no threshold optimization on test predictions**.
- **GC-bias correction**: LOESS (tricube-weighted local regression) fit of per-bin coverage vs GC fraction, applied before classification.
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
