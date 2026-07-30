"""STAGE 6 — SUMMARY.

Reports fit quality against both populations explicitly (total fit
attempts vs. the Stage-2-eligible subset used for the joint-distribution
model), plus a SCREENING section summarizing what was excluded and why.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from . import config


def write_summary(fits: pd.DataFrame, fits_for_stage2: pd.DataFrame,
                  corr: pd.DataFrame, dists: dict, duration: int,
                  screening_counts: dict, decomp: pd.DataFrame | None = None):
    """
    Parameters
    ----------
    decomp : pd.DataFrame or None, default None
        Output of correlation.decompose_correlation() for space in
        {"ab", "munu"}, concatenated. When provided, adds a "CORRELATION
        DECOMPOSITION" section. Default None reproduces the original
        summary.txt exactly.
    """
    lines = []
    w = lines.append
    w("=" * 70)
    w("STOCHASTIC COST PHASING MODEL — RUN SUMMARY")
    w("=" * 70)
    w("")
    w("MODEL: %cost = BetaCDF(%schedule; α, β) fit per project x cost type.")
    w("The Beta(α,β) is interpretable as the distribution of WHEN a dollar")
    w("is spent across normalized schedule.")
    w("")

    w("SCREENING:")
    w(f"  {screening_counts['n_total_groups']} project x cost-type curves loaded")
    w(f"    excluded (monotonicity violation): "
      f"{screening_counts['n_monotonicity_violation']}")
    w(f"    excluded (insufficient interior points): "
      f"{screening_counts['n_insufficient_interior_points']}")
    w(f"    excluded (curve_fit optimizer failure): "
      f"{screening_counts['n_fit_failed']}")
    w(f"    excluded (bound-contact fit, kept in fits.csv): "
      f"{screening_counts['n_bound_contact']}")
    w(f"    clean, used for joint-distribution modeling: "
      f"{screening_counts['n_clean']}")
    if screening_counts["excluded_pairs"]:
        w("  Excluded curves:")
        for proj, ct, reason in screening_counts["excluded_pairs"]:
            w(f"    {proj}/{ct}: {reason}")
    w("  (Full detail in screening_report.csv)")
    w("")

    w("FITS (Stage-2-eligible curves only — bound-contact fits excluded):")
    for ct in sorted(set(fits["cost_type"]) | set(fits_for_stage2["cost_type"])):
        all_ct = fits[fits["cost_type"] == ct]
        clean_ct = fits_for_stage2[fits_for_stage2["cost_type"] == ct]
        n_total = len(all_ct)
        n_insufficient = int((all_ct["status"] == "insufficient_points").sum())
        n_failed = int((all_ct["status"] == "fit_failed").sum())
        n_bound = int(((all_ct["status"] == "fit") & (all_ct["at_bound"])).sum())
        n_used = len(clean_ct)
        w(f"  {ct}: {n_total} total fit attempts; "
          f"{n_insufficient} excluded (insufficient points); "
          f"{n_failed} excluded (fit failed); "
          f"{n_bound} excluded (bound contact); "
          f"{n_used} used for joint-distribution modeling")
        if n_used:
            w(f"       median R2={clean_ct['r2'].median():.4f}  "
              f"mean RMSE={clean_ct['rmse'].mean():.4f}")
            w(f"       α: mean={clean_ct['alpha'].mean():.3f} "
              f"sd={clean_ct['alpha'].std():.3f}   "
              f"β: mean={clean_ct['beta'].mean():.3f} "
              f"sd={clean_ct['beta'].std():.3f}")
            w(f"       mean spend timing (α/(α+β)): "
              f"{clean_ct['mean_timing'].mean():.3f}")
    w("")
    w("CORRELATION (α vs β, Stage-2-eligible curves):")
    w(corr.to_string(index=False))
    if "vintage" in fits_for_stage2.columns and fits_for_stage2["vintage"].notna().any():
        w("")
        w("TEMPORAL DRIFT CHECK (parameters vs project start year, Spearman):")
        for ct, g in fits_for_stage2.groupby("cost_type"):
            g = g.dropna(subset=["vintage"])
            if g["vintage"].nunique() > 2:
                r_a, p_a = spearmanr(g["vintage"], g["alpha"])
                r_b, p_b = spearmanr(g["vintage"], g["beta"])
                r_t, p_t = spearmanr(g["vintage"], g["mean_timing"])
                w(f"  {ct}: α ρ={r_a:+.3f} (p={p_a:.3f})  "
                  f"β ρ={r_b:+.3f} (p={p_b:.3f})  "
                  f"mean timing ρ={r_t:+.3f} (p={p_t:.3f})")
            else:
                w(f"  {ct}: insufficient vintage variation to test")
        w("  (Significant drift would argue for weighting recent projects")
        w("   or adding vintage as a covariate; otherwise pooling is fine.)")
    if decomp is not None and not decomp.empty:
        w("")
        w("CORRELATION DECOMPOSITION (estimator-induced vs population):")
        w("  observed_cov = population_cov + mean(per-curve estimator cov)")
        w("  (method of moments: population_cov_hat = observed_cov -")
        w("  mean(per-curve pcov)). observed_r/estimator_r/population_r below")
        w("  are each computed FROM their own covariance matrix -- unlike the")
        w("  covariances, the correlations themselves are not additive.")
        w("  Beta Fisher information has a strictly negative off-diagonal, so")
        w("  Cov(alpha_hat, beta_hat) > 0 is manufactured by the estimator on")
        w("  top of any true population correlation.")
        w("")
        w(decomp.to_string(index=False))
        for _, r in decomp.iterrows():
            if r["non_pd_flag"]:
                w(f"  WARNING [{r['space']}] {r['cost_type']}: subtracting the "
                  f"estimator component left a non-PD matrix -- population_r "
                  f"is not identified from n={r['n']} curves; falling back to "
                  f"the uncorrected observed_r={r['fallback_r']:+.3f}.")
    w("")
    w("RECOMMENDED PREDICTIVE MODEL: bvn_log (bivariate lognormal)")
    w("  Sample (ln α, ln β) ~ BVN(μ, Σ); exponentiate; evaluate Beta CDF at")
    w("  t = month/duration for any stochastic duration draw.")
    w("")
    for p in dists["bvn_log"]:
        w(f"  {p['cost_type']}: μ_ln=({p['mu'][0]:.4f}, {p['mu'][1]:.4f})  "
          f"σ_ln=({p['sigma'][0]:.4f}, {p['sigma'][1]:.4f})  ρ={p['rho']:+.4f}")
    w("")
    w(f"Example forecast generated for {config.PRIMARY_COST_TYPE}, "
      f"{duration}-month duration (05_tpc_prediction.png).")
    w("")
    w("CAVEATS:")
    w("  - Parameter-distribution estimates carry sampling uncertainty not")
    w("    reflected in the fitted BVN (a hierarchical Bayes extension would")
    w("    propagate it).")
    w("  - Fits assume the Beta CDF family is adequate; check curve overlays")
    w("    and max_abs_err in fits.csv for systematic misfit.")
    w("  - Bound-contact fits are excluded from the joint-distribution model")
    w("    but retained in fits.csv/01_curve_overlays.png for manual review —")
    w("    a curve near a fit bound often signals a step-like or truncated")
    w("    profile the Beta family can't represent well.")

    text = "\n".join(lines)
    print(config.OUTPUT_DIR)
    (config.OUTPUT_DIR / config.SUMMARY_FILENAME).write_text(text, encoding = "utf-8")
    print("\n" + text)
