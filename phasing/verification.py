"""Cross-parameterization verification: (alpha, beta) direct fit vs.
(logit mu, log nu) unconstrained fit.

Confirms the mean-precision reparameterization is a pure change of
coordinates (fitted alpha, beta and curve values agree, up to optimizer
tolerance) rather than a different model, and reports whether it changes
anything about the bound-contact population `phasing.screening` already
tracks: a bound-contact (alpha, beta) fit that lands away from FIT_BOUNDS
under the unconstrained (logit mu, log nu) fit indicates the box
constraint -- not a genuine boundary optimum -- was responsible.

Runs against the full `status == "fit"` population (bound-contact curves
included), not `fits_for_stage2` -- bound-contact resolution is exactly
what this module exists to check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .fitting import beta_cdf


def verify_parameterizations(fits: pd.DataFrame,
                             fits_munu: pd.DataFrame) -> pd.DataFrame:
    """Verify the (alpha,beta) and (mu,nu) fits agree, per curve.

    Checks, per project x cost_type (restricted to `fits` rows with
    status == "fit", i.e. everything `fit_all_curves` actually fit,
    bound-contact included):
      - fitted alpha, beta agree between the two optimizers (rtol=1e-4)
      - the fitted curve values agree on a dense grid (max abs diff < 1e-6)
      - the covariance condition number in each fit's own native
        coordinates: the constrained (alpha,beta) fit's pcov vs. the
        unconstrained (logit mu, log nu) fit's own pcov
      - whether an (alpha,beta) fit that hit FIT_BOUNDS resolves (the
        unconstrained mu,nu fit lands away from the box) or is genuine

    Requires `fits` from `fitting.fit_all_curves(df, keep_mu_nu=True)`.
    """
    fitted = fits[fits["status"] == "fit"]
    merged = fitted.merge(
        fits_munu, on=["project", "cost_type"], suffixes=("_ab", "_munu")
    )
    if merged.empty:
        print("  WARNING: no curves in common between the two fits — "
              "nothing to verify.")
        return merged

    t_grid = np.linspace(0.0, 1.0, 400)
    rows = []
    for _, r in merged.iterrows():
        curve_ab = beta_cdf(t_grid, r["alpha_ab"], r["beta_ab"])
        curve_munu = beta_cdf(t_grid, r["alpha_munu"], r["beta_munu"])
        max_curve_diff = float(np.max(np.abs(curve_ab - curve_munu)))
        alpha_agree = bool(np.allclose(r["alpha_ab"], r["alpha_munu"], rtol=1e-4))
        beta_agree = bool(np.allclose(r["beta_ab"], r["beta_munu"], rtol=1e-4))

        # Compare each optimizer's own covariance in its native
        # coordinates -- raw (mu, nu) units aren't comparable to
        # (alpha, beta) units (mu in (0,1) vs nu up to ~50), so that
        # comparison would be dominated by scale, not conditioning.
        cov_ab = np.array([[r["se_alpha"] ** 2, r["cov_alpha_beta"]],
                           [r["cov_alpha_beta"], r["se_beta"] ** 2]])
        cov_mn = np.asarray(r["pcov_munu"], dtype=float)
        with np.errstate(all="ignore"):
            cond_ab = float(np.linalg.cond(cov_ab))
            cond_mn = float(np.linalg.cond(cov_mn))

        was_at_bound = bool(r["at_bound"])
        resolved = was_at_bound and not (
            np.any(np.isclose(
                [r["alpha_munu"], r["beta_munu"]], config.FIT_BOUNDS[0], atol=1e-2))
            or np.any(np.isclose(
                [r["alpha_munu"], r["beta_munu"]], config.FIT_BOUNDS[1], atol=1e-2))
        )

        rows.append(dict(
            project=r["project"], cost_type=r["cost_type"],
            alpha_agree=alpha_agree, beta_agree=beta_agree,
            max_curve_diff=max_curve_diff,
            cond_cov_ab=cond_ab, cond_cov_munu=cond_mn,
            was_at_bound=was_at_bound, resolved_off_bound=resolved,
        ))

    report = pd.DataFrame(rows)

    print("\nCross-parameterization verification (alpha,beta direct fit vs "
          "mu,nu-space fit):")
    n_ok = int((report["alpha_agree"] & report["beta_agree"]).sum())
    print(f"  {n_ok}/{len(report)} curves: alpha, beta agree to rtol=1e-4")
    print(f"  max curve-value discrepancy across all curves: "
          f"{report['max_curve_diff'].max():.2e} "
          f"(threshold 1e-6)")
    print(f"  median cond(Cov) — constrained (alpha,beta) fit: "
          f"{report['cond_cov_ab'].median():,.1f}   "
          f"unconstrained (logit mu, log nu) fit: "
          f"{report['cond_cov_munu'].median():,.1f}")

    resolved = report[report["was_at_bound"] & report["resolved_off_bound"]]
    still_bound = report[report["was_at_bound"] & ~report["resolved_off_bound"]]
    if len(resolved):
        print(f"  {len(resolved)} bound-contact curve(s) RESOLVED under the "
              "unconstrained mu,nu fit (converged away from FIT_BOUNDS): "
              + ", ".join(f"{p}/{c}" for p, c in
                          zip(resolved["project"], resolved["cost_type"])))
    if len(still_bound):
        print(f"  {len(still_bound)} bound-contact curve(s) remain at the "
              "boundary under the unconstrained fit too (genuine boundary "
              "optimum, not a box artifact): "
              + ", ".join(f"{p}/{c}" for p, c in
                          zip(still_bound["project"], still_bound["cost_type"])))
    if not len(resolved) and not len(still_bound):
        print("  (no curves hit FIT_BOUNDS in the direct fit — nothing to "
              "resolve)")

    print(report.to_string(index=False))
    return report
