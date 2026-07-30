"""Configuration constants shared across the phasing analysis pipeline.

Edit here if your column names differ or you want to change fit/plot behavior.
This module must not import anything else from `phasing` — every other module
depends on it, directly or transitively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# `config.py` lives in `phasing/`, so go up two levels to reach the repo root
# where `data.csv` and `outputs/` are expected to live (this mirrors the old
# single-file script's `Path(__file__).resolve().parent`).
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = SCRIPT_DIR / "data.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Set these to exact column names to skip auto-detection, e.g. "Project"
COL_OVERRIDES = {
    "project": None,      # e.g. "project"
    "cost_type": None,    # e.g. "cost_type"
    "pct_sched": None,    # e.g. "cum_pct_schedule"
    "pct_cost": None,     # e.g. "cum_pct_cost"
    "year": None,         # optional; chronological sort + vintage drift check
}

PRIMARY_COST_TYPE = "TPC"   # stratum used for the headline predictor
COST_TYPE_ORDER = ["TPC", "TEC", "OPC"]  # plot ordering; unknown types appended

FIT_BOUNDS = ([0.05, 0.05], [50.0, 50.0])   # (alpha, beta) search box
MIN_INTERIOR_POINTS = 4                      # min points strictly inside (0,1)x(0,1)
N_PRED_SAMPLES = 10_000                      # Monte Carlo draws for bands
RNG_SEED = 42

# Tolerance for the monotonicity screen: schedule/cost fractions may only
# decrease by float noise up to this amount before a curve is excluded.
MONOTONICITY_TOL = 1e-9

# Output filenames, kept in one place so screening.py and pipeline.py agree.
FITS_FILENAME = "fits.csv"
SCREENING_REPORT_FILENAME = "screening_report.csv"
CORRELATIONS_FILENAME = "correlations.csv"
DISTRIBUTIONS_FILENAME = "distributions.json"
TPC_QUANTILE_TABLE_FILENAME = "tpc_quantile_table.csv"
SUMMARY_FILENAME = "summary.txt"

COLORS = {"TPC": "#1f3864", "TEC": "#2e8b8b", "OPC": "#c9a227"}  # navy/teal/gold
FALLBACK_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
