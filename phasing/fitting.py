"""STAGE 2 — BETA CDF FITTING (per project x cost type)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import betainc

from . import config


def beta_cdf(t, a, b):
    return betainc(a, b, np.clip(t, 0.0, 1.0))


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


def fit_all_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Fit a Beta CDF to every (project, cost_type) group in `df`.

    Every group produces exactly one row, tagged with `status`:
      - "insufficient_points": fewer than MIN_INTERIOR_POINTS interior
        points; alpha/beta/etc are NaN.
      - "fit": curve_fit converged; alpha/beta/etc populated, `at_bound`
        flags whether the solution sits on the FIT_BOUNDS edge.

    No group is silently dropped here — exclusion from downstream stages
    is `phasing.screening`'s job, not this module's.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: BETA CDF FITTING (per project x cost type)")
    print("=" * 70)

    rows = []
    for (proj, ct), g in df.groupby(["project", "cost_type"]):
        t = g["pct_sched"].values.astype(float)
        y = g["pct_cost"].values.astype(float)

        # Drop duplicate schedule points (keep last) and boundary points
        _, keep = np.unique(t, return_index=True)
        t, y = t[keep], y[keep]
        interior = (t > 1e-6) & (t < 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6)
        t_fit, y_fit = t[interior], y[interior]

        if len(t_fit) < config.MIN_INTERIOR_POINTS:
            print(f"  SKIP {proj}/{ct}: only {len(t_fit)} interior points")
            rows.append(dict(
                project=proj, cost_type=ct, status="insufficient_points",
                vintage=(float(g["year"].min())
                         if "year" in g.columns else np.nan),
                alpha=np.nan, beta=np.nan, se_alpha=np.nan, se_beta=np.nan,
                r2=np.nan, rmse=np.nan, max_abs_err=np.nan,
                n_points=len(t), n_fit=len(t_fit), at_bound=False,
                mean_timing=np.nan, mode_timing=np.nan,
            ))
            continue

        p0 = _moment_init(t_fit, y_fit)
        try:
            popt, pcov = curve_fit(
                beta_cdf, t_fit, y_fit, p0=p0, bounds=config.FIT_BOUNDS, maxfev=20000
            )
        except Exception as e:
            print(f"  FAIL {proj}/{ct}: {str(e)[:60]}")
            rows.append(dict(
                project=proj, cost_type=ct, status="insufficient_points",
                vintage=(float(g["year"].min())
                         if "year" in g.columns else np.nan),
                alpha=np.nan, beta=np.nan, se_alpha=np.nan, se_beta=np.nan,
                r2=np.nan, rmse=np.nan, max_abs_err=np.nan,
                n_points=len(t), n_fit=len(t_fit), at_bound=False,
                mean_timing=np.nan, mode_timing=np.nan,
            ))
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

        rows.append(
            dict(
                project=proj, cost_type=ct, status="fit",
                vintage=(float(g["year"].min())
                         if "year" in g.columns else np.nan),
                alpha=a_hat, beta=b_hat, se_alpha=se_a, se_beta=se_b,
                r2=r2, rmse=rmse, max_abs_err=max_abs_err,
                n_points=len(t), n_fit=len(t_fit), at_bound=at_bound,
                # Interpretive stats of the implied spend-timing Beta:
                mean_timing=a_hat / (a_hat + b_hat),
                mode_timing=(
                    (a_hat - 1) / (a_hat + b_hat - 2)
                    if a_hat > 1 and b_hat > 1 else np.nan
                ),
            )
        )

    fits = pd.DataFrame(rows)
    if fits.empty or (fits["status"] == "fit").sum() == 0:
        print("ERROR: no curves fitted. Check data.")
        raise SystemExit(1)

    fitted = fits[fits["status"] == "fit"]
    print(f"\nFitted {len(fitted)} curves "
          f"({(fits['status'] == 'insufficient_points').sum()} skipped)")
    for ct, g in fitted.groupby("cost_type"):
        print(f"  {ct}: n={len(g)}, median R2={g['r2'].median():.4f}, "
              f"mean RMSE={g['rmse'].mean():.4f}, at_bound={g['at_bound'].sum()}")
    nb = fitted["at_bound"].sum()
    if nb:
        print(f"  NOTE: {nb} fits hit parameter bounds — excluded from "
              f"Stage 2, kept in fits.csv for review.")
    return fits
