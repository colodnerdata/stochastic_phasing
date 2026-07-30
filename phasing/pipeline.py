"""Pipeline orchestration: argparse, seeding, and the stage sequence.

Screening is two-phase, inserted around `fitting.fit_all_curves`:
monotonicity is knowable pre-fit (screened before fitting is attempted),
while bound-contact is only knowable post-fit (screened after).
"""

from __future__ import annotations

import argparse

import numpy as np

from . import config, correlation, data, distributions, eda, fitting, screening, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=24,
                    help="Example forecast duration in months (default 24)")
    ap.add_argument("--model", choices=["bvn_log", "bvn_unit"],
                    default="bvn_log", help="Predictor sampling model")
    args = ap.parse_args()

    np.random.seed(config.RNG_SEED)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    df, _ = data.load_and_validate()

    df_clean, pre_fit_exclusions = screening.screen_pre_fit(df)

    fits = fitting.fit_all_curves(df_clean)
    fits.to_csv(config.OUTPUT_DIR / config.FITS_FILENAME, index=False)
    print(f"Saved -> {config.OUTPUT_DIR / config.FITS_FILENAME}")

    fits_for_stage2, post_fit_exclusions = screening.screen_post_fit(fits)

    report_df = screening.build_screening_report(
        df, pre_fit_exclusions, fits, post_fit_exclusions
    )
    screening.write_screening_report(
        report_df, config.OUTPUT_DIR / config.SCREENING_REPORT_FILENAME
    )
    screening_counts = screening.screening_summary_counts(report_df)

    print("\n" + "=" * 70)
    print("STAGE 3: EDA PLOTS")
    print("=" * 70)
    eda.plot_curve_overlays(df_clean, fits)
    eda.plot_scatter_ellipses(fits)
    print("Saved -> 01_curve_overlays.png, 02_scatter_ellipses.png")

    corr = correlation.correlation_analysis(fits_for_stage2)
    correlation.plot_correlation_panels(fits_for_stage2)
    correlation.plot_qq_normality(fits_for_stage2)
    print("Saved -> 03_correlation_panels.png, 04_qq_normality.png")

    dists = distributions.fit_joint_distributions(fits_for_stage2)

    if config.PRIMARY_COST_TYPE in fits_for_stage2["cost_type"].values:
        pred = distributions.PhasingPredictor(dists, model=args.model)
        distributions.plot_prediction(pred, config.PRIMARY_COST_TYPE, args.duration)
        qt = pred.quantile_table(config.PRIMARY_COST_TYPE, args.duration)
        qt.to_csv(config.OUTPUT_DIR / config.TPC_QUANTILE_TABLE_FILENAME, index=False)
        print(f"Saved -> 05_tpc_prediction.png, {config.TPC_QUANTILE_TABLE_FILENAME}")
    else:
        print(f"WARNING: {config.PRIMARY_COST_TYPE} not in Stage-2-eligible "
              f"data; skipping prediction.")

    summary.write_summary(fits, fits_for_stage2, corr, dists, args.duration,
                          screening_counts)
    print("\nDONE. All outputs in:", config.OUTPUT_DIR)
