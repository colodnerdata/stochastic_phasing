"""STAGE 2 — BETA CDF FITTING (per project x cost type)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import betainc, expit

from . import config
from .parameterization import ab_to_mu_nu, delta_method_cov, mu_nu_to_ab


def beta_cdf(t, a, b):
    return betainc(a, b, np.clip(t, 0.0, 1.0))


_MU_NU_NAN_FIELDS = dict(
    mu=np.nan, nu=np.nan, se_mu=np.nan, se_nu=np.nan,
    cov_mu_nu=np.nan, cov_alpha_beta=np.nan,
    logit_mu=np.nan, log_nu=np.nan,
)


def _moment_init(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Method-of-moments initializer.
    Treat spend increments dy as probability mass at interval midpoints of t:
    the Beta(a,b) being fitted is the distribution of 'when a dollar is spent'.
      m = sum(mid * dy),  v = sum((mid-m)^2 * dy)
      k = m(1-m)/v - 1;  a = m*k, b = (1-m)*k
    """
    tt = np.concatenate([[0.0], t, [1.0]])
    yy = np.concatenate([[0.0], y, [1.0]])
    dy = np.diff(yy)
    mid = 0.5 * (tt[:-1] + tt[1:])
    dy = np.maximum(dy, 0)
    if dy.sum() <= 0:
        return 2.0, 2.0
    dy = dy / dy.sum()
    m = float(np.sum(mid * dy))
    v = float(np.sum((mid - m) ** 2 * dy))
    m = min(max(m, 1e-3), 1 - 1e-3)
    if v <= 1e-8:
        return 2.0, 2.0
    k = m * (1 - m) / v - 1
    if k <= 0:
        return 1.0, 1.0
    a, b = m * k, (1 - m) * k
    lo, hi = config.FIT_BOUNDS[0][0], config.FIT_BOUNDS[1][0]
    return float(np.clip(a, lo, hi)), float(np.clip(b, lo, hi))


def fit_all_curves(df: pd.DataFrame, keep_mu_nu: bool = False) -> pd.DataFrame:
    """Fit a Beta CDF to every (project, cost_type) group in `df`.

    Every group produces exactly one row, tagged with `status`:
      - "insufficient_points": fewer than MIN_INTERIOR_POINTS interior
        points; alpha/beta/etc are NaN.
      - "fit": curve_fit converged; alpha/beta/etc populated, `at_bound`
        flags whether the solution sits on the FIT_BOUNDS edge.

    No group is silently dropped here — exclusion from downstream stages
    is `phasing.screening`'s job, not this module's.

    Parameters
    ----------
    keep_mu_nu : bool, default False
        If True, also attach the mean-precision (mu, nu) reparameterization
        and its delta-method standard errors/covariance (from `pcov`) to
        "fit" rows; NaN-filled for "insufficient_points"/"fit_failed" rows.
        Default False reproduces the original column set exactly.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: BETA CDF FITTING (per project x cost type)")
    print("=" * 70)

    rows = []
    for (proj, ct), g in df.groupby(["project", "cost_type"]):
        t = g["pct_sched"].values.astype(float)
        y = g["pct_cost"].values.astype(float)

        # Drop duplicate schedule points (keep last) and boundary points.
        # np.unique keeps the FIRST occurrence of each value, so reverse
        # both arrays first to make that first occurrence the last one
        # in original (ascending-t) order.
        t_rev, y_rev = t[::-1], y[::-1]
        _, keep_rev = np.unique(t_rev, return_index=True)
        t, y = t_rev[keep_rev], y_rev[keep_rev]
        interior = (t > 1e-6) & (t < 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6)
        t_fit, y_fit = t[interior], y[interior]

        if len(t_fit) < config.MIN_INTERIOR_POINTS:
            print(f"  SKIP {proj}/{ct}: only {len(t_fit)} interior points")
            row = dict(
                project=proj, cost_type=ct, status="insufficient_points",
                fail_reason=None,
                vintage=(float(g["year"].min())
                         if "year" in g.columns else np.nan),
                alpha=np.nan, beta=np.nan, se_alpha=np.nan, se_beta=np.nan,
                r2=np.nan, rmse=np.nan, max_abs_err=np.nan,
                n_points=len(t), n_fit=len(t_fit), at_bound=False,
                mean_timing=np.nan, mode_timing=np.nan,
            )
            if keep_mu_nu:
                row.update(_MU_NU_NAN_FIELDS)
            rows.append(row)
            continue

        p0 = _moment_init(t_fit, y_fit)
        try:
            popt, pcov = curve_fit(
                beta_cdf, t_fit, y_fit, p0=p0, bounds=config.FIT_BOUNDS, maxfev=20000
            )
        except Exception as e:
            print(f"  FAIL {proj}/{ct}: {str(e)[:60]}")
            row = dict(
                project=proj, cost_type=ct, status="fit_failed",
                fail_reason=str(e)[:200],
                vintage=(float(g["year"].min())
                         if "year" in g.columns else np.nan),
                alpha=np.nan, beta=np.nan, se_alpha=np.nan, se_beta=np.nan,
                r2=np.nan, rmse=np.nan, max_abs_err=np.nan,
                n_points=len(t), n_fit=len(t_fit), at_bound=False,
                mean_timing=np.nan, mode_timing=np.nan,
            )
            if keep_mu_nu:
                row.update(_MU_NU_NAN_FIELDS)
            rows.append(row)
            continue

        a_hat, b_hat = popt
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.diag(pcov))
        se_a, se_b = (se if np.all(np.isfinite(se)) else (np.nan, np.nan))

        resid = y_fit - beta_cdf(t_fit, a_hat, b_hat)
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
        r2 = 1 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else np.nan
        max_abs_err = float(np.max(np.abs(resid)))

        at_bound = (
            np.any(np.isclose(popt, config.FIT_BOUNDS[0], atol=1e-3))
            or np.any(np.isclose(popt, config.FIT_BOUNDS[1], atol=1e-3))
        )

        row = dict(
            project=proj, cost_type=ct, status="fit",
            fail_reason=None,
            vintage=(float(g["year"].min())
                     if "year" in g.columns else np.nan),
            alpha=a_hat, beta=b_hat, se_alpha=se_a, se_beta=se_b,
            r2=r2, rmse=rmse, max_abs_err=max_abs_err,
            n_points=len(t), n_fit=len(t_fit), at_bound=at_bound,
            # Interpretive stats of the implied spend-timing Beta:
            # mean_timing is kept for backward compatibility; it equals mu
            # below and mu is the canonical mean-precision mean parameter.
            mean_timing=a_hat / (a_hat + b_hat),
            mode_timing=(
                (a_hat - 1) / (a_hat + b_hat - 2)
                if a_hat > 1 and b_hat > 1 else np.nan
            ),
        )

        if keep_mu_nu:
            mu_hat, nu_hat = ab_to_mu_nu(a_hat, b_hat)
            cov_ab = (pcov if np.all(np.isfinite(pcov))
                      else np.full((2, 2), np.nan))
            cov_mn = delta_method_cov(cov_ab, a_hat, b_hat)
            row.update(dict(
                mu=mu_hat, nu=nu_hat,
                se_mu=np.sqrt(cov_mn[0, 0]), se_nu=np.sqrt(cov_mn[1, 1]),
                cov_mu_nu=cov_mn[0, 1], cov_alpha_beta=cov_ab[0, 1],
                logit_mu=np.log(mu_hat / (1.0 - mu_hat)), log_nu=np.log(nu_hat),
            ))

        rows.append(row)

    fits = pd.DataFrame(rows)
    if fits.empty or (fits["status"] == "fit").sum() == 0:
        print("ERROR: no curves fitted. Check data.")
        raise SystemExit(1)

    fitted = fits[fits["status"] == "fit"]
    n_insufficient = (fits["status"] == "insufficient_points").sum()
    n_failed = (fits["status"] == "fit_failed").sum()
    print(f"\nFitted {len(fitted)} curves "
          f"({n_insufficient} skipped for insufficient points, "
          f"{n_failed} fit failures)")
    for ct, g in fitted.groupby("cost_type"):
        print(f"  {ct}: n={len(g)}, median R2={g['r2'].median():.4f}, "
              f"mean RMSE={g['rmse'].mean():.4f}, at_bound={g['at_bound'].sum()}")
    nb = fitted["at_bound"].sum()
    if nb:
        print(f"  NOTE: {nb} fits hit parameter bounds — excluded from "
              f"Stage 2, kept in fits.csv for review.")
    return fits


def _munu_beta_cdf(t, logit_mu, log_nu):
    """Beta CDF reparameterized by unconstrained (logit mu, log nu)."""
    mu = expit(logit_mu)
    nu = np.exp(log_nu)
    alpha, beta = mu_nu_to_ab(mu, nu)
    return beta_cdf(t, alpha, beta)


def fit_all_curves_mu_nu(df: pd.DataFrame) -> pd.DataFrame:
    """Refit every curve in unconstrained (logit mu, log nu) space.

    This is a verification fit, not a replacement (see
    `phasing.verification.verify_parameterizations`): optimizing over
    (logit mu, log nu) needs no bounds (mu is squashed to (0,1) via expit,
    nu is squashed to (0,inf) via exp), so it also tells us whether curves
    that hit the (alpha, beta) FIT_BOUNDS box do so because that is the
    genuine optimum or because of the artificial box constraint.

    Uses the same duplicate-handling / interior-point selection as
    `fit_all_curves`, but skips (rather than status-tags) curves it can't
    fit -- this is a diagnostic utility, not a primary output.

    Returns
    -------
    pd.DataFrame
        One row per successfully fitted curve: project, cost_type, mu, nu,
        alpha, beta (implied), logit_mu, log_nu, and pcov_munu (the raw
        2x2 covariance of (logit_mu_hat, log_nu_hat) from curve_fit).
    """
    rows = []
    for (proj, ct), g in df.groupby(["project", "cost_type"]):
        t = g["pct_sched"].values.astype(float)
        y = g["pct_cost"].values.astype(float)

        t_rev, y_rev = t[::-1], y[::-1]
        _, keep_rev = np.unique(t_rev, return_index=True)
        t, y = t_rev[keep_rev], y_rev[keep_rev]
        interior = (t > 1e-6) & (t < 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6)
        t_fit, y_fit = t[interior], y[interior]

        if len(t_fit) < config.MIN_INTERIOR_POINTS:
            continue

        a0, b0 = _moment_init(t_fit, y_fit)
        mu0, nu0 = ab_to_mu_nu(a0, b0)
        p0 = [np.log(mu0 / (1.0 - mu0)), np.log(nu0)]
        try:
            popt, pcov = curve_fit(
                _munu_beta_cdf, t_fit, y_fit, p0=p0, maxfev=20000
            )
        except Exception as e:
            print(f"  FAIL (mu,nu) {proj}/{ct}: {str(e)[:60]}")
            continue

        logit_mu_hat, log_nu_hat = popt
        mu_hat, nu_hat = expit(logit_mu_hat), np.exp(log_nu_hat)
        alpha_hat, beta_hat = mu_nu_to_ab(mu_hat, nu_hat)

        rows.append(dict(
            project=proj, cost_type=ct,
            mu=mu_hat, nu=nu_hat, alpha=alpha_hat, beta=beta_hat,
            logit_mu=logit_mu_hat, log_nu=log_nu_hat,
            pcov_munu=pcov,
        ))

    return pd.DataFrame(rows)
