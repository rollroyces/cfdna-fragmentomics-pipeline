# cfdna-fragmentomics-pipeline — master Snakemake workflow
#
# Tumor-naive cfDNA fragmentomics: raw BAM (or FinaleDB pre-processed
# fragment records) → FSD + DELFI + WPS features → GC correction →
# Cancer-vs-Healthy classification with rigorous CV.
#
# Run:
#   snakemake --cores 4 -s main.smk --configfile config.yaml
#   snakemake --cores 4 -s main.smk --configfile config.yaml --report report.html

configfile: "config.yaml"

import os

samples = config["samples"]
labels  = config.get("labels", {})
mode    = config.get("mode", "finaledb")   # "finaledb" | "bam"
ref     = config.get("reference_fasta", "")
out_dir = config.get("out_dir", "results")

# ── Rules ──────────────────────────────────────────────────────────────

rule all:
    input:
        os.path.join(out_dir, "classifier_results.json")

# ALIGNMENT (BAM mode: raw FASTQ → BWA-MEM → coordinate sort)
rule align:
    """BWA-MEM alignment of raw FASTQ pairs to hg38 (production path)."""
    output:
        "data/raw/{sample}.aligned.bam"
    params:
        ref=ref,
        threads=config.get("threads", 4),
    run:
        shell(f"bwa mem -t {params.threads} {params.ref} "
              f"data/raw/{wildcards.sample}_R1.fastq.gz "
              f"data/raw/{wildcards.sample}_R2.fastq.gz | "
              f"samtools sort -@ {params.threads} -o {{output}} -")

# DEDUP (Picard MarkDuplicates — required before fragmentomics extraction)
rule dedup:
    """Picard MarkDuplicates: PCR duplicate removal."""
    output:
        "data/raw/{sample}.mdups.bam",
        "data/raw/{sample}.mdups.bam.bai",
    input:
        "data/raw/{sample}.aligned.bam"
    params:
        threads=config.get("threads", 4),
    run:
        shell(f"picard MarkDuplicates I={{input}} O={{output[0]}} "
              f"M=data/raw/{wildcards.sample}.dup_metrics.txt "
              f"REMOVE_DUPLICATES=true ASSUME_SORTED=true && "
              f"samtools index {{output[0]}}")

# FETCH (FinaleDB mode only)
rule fetch_data:
    """Download real cfDNA fragment records from FinaleDB (streaming)."""
    output:
        manifest=os.path.join("data/raw", "manifest.json")
    params:
        disease=config.get("disease", ""),
        healthy=config.get("healthy", False),
        n=config.get("n_samples", 12),
    run:
        cmd = (f"python scripts/fetch_finaledb.py --n {params.n} "
               f"--out-dir data/raw")
        if params.disease:
            cmd += f" --disease '{params.disease}'"
        if params.healthy:
            cmd += " --healthy"
        shell(cmd)

# FSD
rule fsd:
    output:
        "data/features/{sample}.fsd.json"
    params:
        mode=mode
    run:
        if params.mode == "bam":
            shell(f"python scripts/extract_fsd.py --input data/raw/{wildcards.sample}.mdups.bam "
                  f"--mode bam --sample {wildcards.sample} --out-dir data/features")
        else:
            shell(f"python scripts/extract_fsd.py --input data/raw/{wildcards.sample}.frag.tsv.bgz "
                  f"--mode frag --sample {wildcards.sample} --out-dir data/features")

# DELFI + WPS
rule delfi:
    output:
        "data/features/{sample}.delfi.json",
        "data/features/{sample}.delfi_100kb_ratio.npy",
        "data/features/{sample}.wps_100kb.npy",
        "data/features/{sample}.delfi_100kb_counts.npy",
    run:
        shell(f"python scripts/extract_delfi.py --input data/raw/{wildcards.sample}.frag.tsv.bgz "
              f"--sample {wildcards.sample} --out-dir data/features")

# GC correction (needs per-bin GC reference)
rule gc_correct:
    output:
        "data/features/{sample}.gc_corrected.npy"
    input:
        bins="data/features/{sample}.delfi_100kb_ratio.npy",
        counts="data/features/{sample}.delfi_100kb_counts.npy",
        gc=config.get("gc_reference", "data/references/hg38_100kb_gc.npy"),
    run:
        shell(f"python scripts/gc_correction.py --bins {input.bins} "
              f"--counts {input.counts} --gc {input.gc} "
              f"--sample {wildcards.sample} --out-dir data/features")

# Motifs (BAM mode only)
rule motifs:
    output:
        "data/features/{sample}.motifs.json"
    run:
        shell(f"python scripts/extract_motifs.py --bam data/raw/{wildcards.sample}.mdups.bam "
              f"--ref {ref} --sample {wildcards.sample} --out-dir data/features")

# Label table
rule labels:
    output:
        "data/features/labels.tsv"
    run:
        with open(output.labels, "w") as f:
            for s, lab in labels.items():
                f.write(f"{s}\t{lab}\n")

# Classification
rule classify:
    input:
        labels=os.path.join("data/features", "labels.tsv"),
        features=expand("data/features/{s}.fsd.json", s=samples),
    output:
        os.path.join(out_dir, "classifier_results.json")
    params:
        model=config.get("model", "rf"),
        cv=config.get("cv", 5),
        with_motifs=config.get("with_motifs", False),
    run:
        shell(f"python scripts/train_classifier.py --features data/features "
              f"--labels {input.labels} --out {out_dir} "
              f"--model {params.model} --cv {params.cv} "
              f"{'--with-motifs' if params.with_motifs else ''}")
