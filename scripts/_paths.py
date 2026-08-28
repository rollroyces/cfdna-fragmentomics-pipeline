"""Shared constants for cfDNA fragmentomics pipeline scripts.

Centralizes hardcoded paths so the scripts work no matter where the
repo is checked out. All scripts should import from here:
  from _paths import REPO_ROOT, FEAT_DIR, LABELS_TSV, SCRIPT_DIR

REPO_ROOT is computed relative to __file__ so the scripts work
from any directory, including when run via `python -m scripts.X`.
"""
from __future__ import annotations

from pathlib import Path

# Layout:
#   <REPO_ROOT>/
#     data/features/{sample}.{delfi_5mb_ratio,delfi_5mb_coverage,...}.npy
#     data/features/labels.tsv
#     data/features/labels_cross_study.tsv
#     scripts/{honest_benchmark.py, lr_no_pca_vs_pca200.py, ...}
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
FEAT_DIR = DATA_DIR / "features"
LABELS_TSV = FEAT_DIR / "labels.tsv"
LABELS_CROSS_STUDY_TSV = FEAT_DIR / "labels_cross_study.tsv"
RESULTS_DIR = REPO_ROOT / "results"

# Default Gemma model path (Apple Silicon local model)
DEFAULT_GEMMA_MODEL_PATH = Path.home() / "models" / "gemma-2-9b-it-Q4_K_M.gguf"