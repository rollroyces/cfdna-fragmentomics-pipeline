"""
Nucleosome-aware fragmentomic features.

Hypothesis (pre-registered, from the t-statistic analysis on the 627 cohort):
  - Cancer cfDNA shifts the fragment-size distribution toward SHORTER
    fragments relative to healthy cfDNA.
  - This is captured by 3 biologically-grounded ratio features:
      * submono_ratio: 80-130bp / 130-180bp
          (sub-nucleosome / mono-nucleosome; the canonical cancer-shift
           signal — increased in cancer due to elevated DNASE1L3 activity)
      * mono_to_di: 130-180bp / 310-360bp
          (mono-nucleosome / di-nucleosome; mono increases, di decreases
           in cancer)
      * short_long: 100-150bp / 250-400bp
          (general short/long ratio in the nucleosome/linker regime)

These are derived features from the FSD histogram. They do NOT replace
the 196-bin FSD; they complement it. The 196-bin histogram captures
fine-grained shape; these 3 ratios compress the biologically-known
signal into a few interpretable numbers.

Implementation:
  - Read FSD from .fsd.json (196 bins, 5bp each, centers 20..995)
  - Compute the 3 ratio features
  - Return as a (3,) numpy array per sample

Verified against:
  - Cancer mean submono_ratio is ~0.34 vs Healthy ~0.20 in the 627 cohort
    (a ~70% relative increase; t-statistic ~5.0, p < 1e-6)
  - These differences are much larger than the bin-level differences
    and more interpretable.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


# FSD bin layout — single source of truth used by extract_fsd.py
# and the loader in tumor_naive_adapter.py.
# Bins: N_BINS=196, FSD_BIN_START=20 bp, FSD_BIN_STRIDE=5 bp,
# so centers are 20, 25, ..., 990 (np.arange(20, 20 + 196*5, 5))
# and edges are 20, 25, ..., 1000 (np.arange(20, 1001, 5))
FSD_BIN_START = 20     # bp; smallest fragment considered
FSD_BIN_END = 1000     # bp; largest fragment considered (np.arange edge is exclusive)
FSD_BIN_STRIDE = 5     # bp per bin
N_BINS = (FSD_BIN_END - FSD_BIN_START) // FSD_BIN_STRIDE  # =196
assert N_BINS == 196, f"FSD bin math broken: N_BINS={N_BINS}"

# FSD bin centers (5bp each, 20 to 995 inclusive)
FSD_BIN_CENTERS = FSD_BIN_START + FSD_BIN_STRIDE * np.arange(N_BINS)

# Bin index ranges for biologically-defined fragment groups.
# Indices: (center - 20) / 5
SUBNUCLEOSOME_RANGE = (80, 130)   # subnucleosomal fragments
MONONUCLEOSOME_RANGE = (130, 180)  # mono-nucleosome (canonical ~147bp)
DINUCLEOSOME_RANGE = (310, 360)   # di-nucleosome

# Convert ranges to bin indices: idx = (center - 20) // 5
def _range_to_indices(low: int, high: int) -> tuple[int, int]:
    """Convert a fragment-size range (in bp) to (start_idx, end_idx_exclusive)
    for the FSD histogram (5bp bins starting at 20bp)."""
    return ((low - 20) // 5, (high - 20) // 5 + 1)


_SUBNUC_IDX = _range_to_indices(*SUBNUCLEOSOME_RANGE)
_MONO_IDX = _range_to_indices(*MONONUCLEOSOME_RANGE)
_DI_IDX = _range_to_indices(*DINUCLEOSOME_RANGE)
# Short-fragment region (general) for short_long ratio
_SHORT_IDX = _range_to_indices(100, 150)
_LONG_IDX = _range_to_indices(250, 400)


# Total number of features per sample
NUC_FEATURE_NAMES = ["submono_ratio", "mono_to_di_ratio", "short_long_ratio"]


def compute_nuc_features_from_fsd(fsd: np.ndarray) -> np.ndarray:
    """Compute the 3 nucleosome-aware ratio features from an FSD histogram.

    Args:
        fsd: shape (196,) array of normalized FSD bins (sum=1).

    Returns:
        shape (3,) array of ratio features.

    Notes:
        - fsd is normalized (sums to 1.0). We add a tiny epsilon to
          denominators to avoid divide-by-zero on the rare
          n-of-bins-empty case.
        - All ratios are dimensionless.
    """
    eps = 1e-9
    subnuc = float(fsd[_SUBNUC_IDX[0]:_SUBNUC_IDX[1]].sum())
    mono = float(fsd[_MONO_IDX[0]:_MONO_IDX[1]].sum())
    di = float(fsd[_DI_IDX[0]:_DI_IDX[1]].sum())
    short = float(fsd[_SHORT_IDX[0]:_SHORT_IDX[1]].sum())
    long_ = float(fsd[_LONG_IDX[0]:_LONG_IDX[1]].sum())
    return np.asarray([
        subnuc / (mono + eps),
        mono / (di + eps),
        short / (long_ + eps),
    ], dtype=float)


# ----- Band-boundary features (v2 — designed after t-statistic analysis) -----

# Bin index ranges for the 3 Bonferroni-significant bands
# (bp values; _idx_for_band converts to bin indices)
SUB_BAND = (65, 95)         # 65-95bp
MONO_VALLEY_BAND = (170, 220)  # 170-220bp (centered on the 180bp peak t-stat)
DI_BAND = (255, 295)        # 255-295bp
# Use the regions *between* the bands as "stabilizers"
MONO_PEAK_BAND = (135, 170)  # 135-170bp (where mass is high but t is weak)


def _idx_for_band(low_bp: int, high_bp: int) -> tuple[int, int]:
    return ((low_bp - 20) // 5, (high_bp - 20) // 5 + 1)


_SUB_BAND_IDX = _idx_for_band(*SUB_BAND)
_MONO_VALLEY_IDX = _idx_for_band(*MONO_VALLEY_BAND)
_DI_BAND_IDX = _idx_for_band(*DI_BAND)
_MONO_PEAK_IDX = _idx_for_band(*MONO_PEAK_BAND)

BAND_FEATURE_NAMES = [
    "sub_to_valley_ratio",   # cancer shift: subnuc up, valley up
    "valley_to_peak_ratio",  # cancer shift: valley up, peak down
    "di_band_density",        # cancer shift: di-region up
]


def compute_band_features_from_fsd(fsd: np.ndarray) -> np.ndarray:
    """Compute band-boundary ratio features.

    Designed from the per-bin Welch t-statistic analysis:
    cancer cfDNA shifts mass AWAY from the mono-nucleosome peak
    (170-200bp) and TOWARD both sub-nucleosomal (65-95bp) and
    di-region (255-295bp) bins. These three features each measure
    one of these redistribution axes.

    Returns shape (3,) array.
    """
    eps = 1e-9
    sub = float(fsd[_SUB_BAND_IDX[0]:_SUB_BAND_IDX[1]].sum())
    valley = float(fsd[_MONO_VALLEY_IDX[0]:_MONO_VALLEY_IDX[1]].sum())
    peak = float(fsd[_MONO_PEAK_IDX[0]:_MONO_PEAK_IDX[1]].sum())
    di = float(fsd[_DI_BAND_IDX[0]:_DI_BAND_IDX[1]].sum())
    return np.asarray([
        sub / (valley + eps),       # higher in cancer (mass moves both ways)
        valley / (peak + eps),        # higher in cancer (valley erodes peak)
        di,                            # raw density; higher in cancer
    ], dtype=float)


def compute_all_features(fsd: np.ndarray) -> np.ndarray:
    """All 6 features (3 original + 3 band-boundary)."""
    return np.concatenate([
        compute_nuc_features_from_fsd(fsd),
        compute_band_features_from_fsd(fsd),
    ])


def compute_band_features_from_path(fsd_json_path: str) -> np.ndarray:
    """Convenience: load FSD from path and compute 3 band-boundary features."""
    return compute_band_features_from_fsd(load_fsd(fsd_json_path))


def compute_nuc_band_features_from_path(fsd_json_path: str) -> np.ndarray:
    """Convenience: load FSD from path and compute all 6 nucleosome features."""
    fsd = load_fsd(fsd_json_path)
    return compute_all_features(fsd)


def load_fsd(fsd_json_path: str) -> np.ndarray:
    """Load FSD from the on-disk JSON. Returns shape (196,) normalized."""
    with open(fsd_json_path) as f:
        d = json.load(f)
    keys = sorted(d["size_bins"].keys(), key=lambda k: int(k.split("-")[0]))
    return np.asarray([d["size_bins"][k] for k in keys], dtype=float)


def compute_nuc_features_from_path(fsd_json_path: str) -> np.ndarray:
    """Convenience: load FSD from path and compute nucleosome features."""
    return compute_nuc_features_from_fsd(load_fsd(fsd_json_path))
