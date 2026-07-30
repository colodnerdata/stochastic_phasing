"""
Stochastic Cost Phasing Analysis — Pipeline
============================================
Loads `data.csv` from this directory (or the path passed via --data) and
runs:

  1. Load & validate (column resolution, scaling, chronological sort)
  2. Data-quality screening (monotonicity exclusion, min-points, bound-contact)
  3. Beta CDF fit per project x cost type
  4. EDA on fitted (alpha, beta): scatter + 95% confidence ellipses, curve overlays
  5. Correlation (unit & log space) + bivariate normality checks
  6. Joint distribution fits (BVN in unit space, BVN in log space = bivariate lognormal)
  7. PhasingPredictor + example TPC forecast with credible bands

Expected columns (flexible name matching, override in phasing/config.py).
Two input formats are auto-detected, based on which columns are present:
  cumulative:  project, cost_type, year, cum_pct_cost, cum_pct_schedule
  incremental: project, year_from_start, inc_pct_spend, cum_pct_sched
    (no cost_type column; single undifferentiated spend curve per project)

Outputs written to ./outputs/ next to this script:
  fits.csv, screening_report.csv, correlations.csv, distributions.json,
  01_curve_overlays.png, 02_scatter_ellipses.png,
  03_correlation_panels.png, 04_qq_normality.png,
  05_tpc_prediction.png, tpc_quantile_table.csv, summary.txt
  (--param munu/both additionally add mean-precision (mu, nu) columns to
  fits.csv, a bvn_munu entry to distributions.json, and, for 'both', a
  cross-parameterization verification, a correlation decomposition
  (extra columns in correlations.csv + a summary.txt section), and
  06_munu_scatter.png)

Usage:
  python phasing_analysis.py            # full run, TPC as primary stratum
  python phasing_analysis.py --duration 36   # change example forecast duration
  python phasing_analysis.py --param both    # + mean-precision parameterization
  python phasing_analysis.py --data path/to/other.csv   # run against a
                                                          # different dataset

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
