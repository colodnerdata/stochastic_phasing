"""Pipeline orchestration: argparse, seeding, and the stage sequence.

Screening is two-phase, inserted around `fitting.fit_all_curves`:
monotonicity is knowable pre-fit (screened before fitting is attempted),
while bound-contact is only knowable post-fit (screened after).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import (
    config,
    correlation,
    data,
    distributions,
    eda,
    fitting,
    screening,
    summary,
    verification,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=24,
                    help="Example forecast duration in months (default 24)")
    ap.add_argument("--model", choices=["bvn_log", "bvn_unit", "bvn_munu"],
                    default="bvn_log", help="Predictor sampling model "
                    "('bvn_munu' requires --param munu or --param both)")
    ap.add_argument("--param", choices=["ab", "munu", "both"], default="ab",
                    help="Parameterization mode. 'ab' (default) reproduces "
                    "current behavior exactly. 'munu' also adds mean-"
                    "precision (mu, nu) columns to fits.csv and a bvn_munu "
                    "entry to distributions.json. 'both' additionally runs "
                    "the mu,nu-space cross-parameterization verification, "
                    "the estimator-vs-population correlation decomposition, "
                    "and writes 06_munu_scatter.png.")
    args = ap.parse_args()

    np.random.seed(config.RNG_SEED)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    use_munu = args.param in ("munu", "both")

    df, _ = data.load_and_validate()

    df_clean, pre_fit_exclusions = screening.screen_pre_fit(df)

    fits = fitting.fit_all_curves(df_clean, keep_mu_nu=use_munu)
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

    dists = distributions.fit_joint_distributions(fits_for_stage2, include_munu=use_munu)

    decomp = None
    if args.param == "both":
        print("\n" + "=" * 70)
        print("STAGE 2b: MEAN-PRECISION VERIFICATION & DECOMPOSITION")
        print("=" * 70)
        fits_munu = fitting.fit_all_curves_mu_nu(df_clean)
        verification.verify_parameterizations(fits, fits_munu)

        decomp_ab = correlation.decompose_correlation(fits_for_stage2, space="ab")
        decomp_munu = correlation.decompose_correlation(fits_for_stage2, space="munu")
        decomp = pd.concat([decomp_ab, decomp_munu], ignore_index=True)

        decomp_wide = decomp.pivot(index="cost_type", columns="space")
        decomp_wide.columns = [f"{sp}_{col}" for col, sp in decomp_wide.columns]
        decomp_wide = decomp_wide.reset_index()
        corr = corr.merge(decomp_wide, on="cost_type", how="left")
        corr.to_csv(config.OUTPUT_DIR / config.CORRELATIONS_FILENAME, index=False)
        print(f"Saved -> {config.OUTPUT_DIR / config.CORRELATIONS_FILENAME} "
              "(with correlation-decomposition columns)")

        eda.plot_munu_scatter(fits)
        print("Saved -> 06_munu_scatter.png")

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
                          screening_counts, decomp=decomp)
    print("\nDONE. All outputs in:", config.OUTPUT_DIR)
