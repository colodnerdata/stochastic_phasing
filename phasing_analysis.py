"""
Stochastic Cost Phasing Analysis — Pipeline
============================================
Loads `data.csv` from this directory and runs:

  1. Load & validate (column resolution, scaling, chronological sort)
  2. Data-quality screening (monotonicity exclusion, min-points, bound-contact)
  3. Beta CDF fit per project x cost type
  4. EDA on fitted (alpha, beta): scatter + 95% confidence ellipses, curve overlays
  5. Correlation (unit & log space) + bivariate normality checks
  6. Joint distribution fits (BVN in unit space, BVN in log space = bivariate lognormal)
  7. PhasingPredictor + example TPC forecast with credible bands

Expected columns (flexible name matching, override in phasing/config.py):
  project, cost_type, year, cum_pct_cost, cum_pct_schedule

Outputs written to ./outputs/ next to this script:
  fits.csv, screening_report.csv, correlations.csv, distributions.json,
  01_curve_overlays.png, 02_scatter_ellipses.png,
  03_correlation_panels.png, 04_qq_normality.png,
  05_tpc_prediction.png, tpc_quantile_table.csv, summary.txt

Usage:
  python phasing_analysis.py            # full run, TPC as primary stratum
  python phasing_analysis.py --duration 36   # change example forecast duration

Requires: numpy, pandas, scipy, matplotlib

Implementation lives in the `phasing/` package (config, data, screening,
fitting, eda, correlation, distributions, summary, pipeline) — this script
is just the entry point.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from phasing.pipeline import main

if __name__ == "__main__":
    main()
