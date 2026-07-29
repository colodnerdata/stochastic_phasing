"""
Stochastic Cost Phasing Analysis — Monolith Pipeline
=====================================================
Loads `data.csv` from the same directory as this script and runs:

  1. Load & validate (column resolution, scaling, monotonicity)
  2. Beta CDF fit per project x cost type
  3. EDA on fitted (alpha, beta): scatter + 95% confidence ellipses, curve overlays
  4. Correlation (unit & log space) + bivariate normality checks
  5. Joint distribution fits (BVN in unit space, BVN in log space = bivariate lognormal)
  6. PhasingPredictor + example TPC forecast with credible bands

Expected columns (flexible name matching, override in CONFIG below):
  project, cost_type, year, cum_pct_cost, cum_pct_schedule

Outputs written to ./outputs/ next to this script:
  fits.csv, correlations.csv, distributions.json,
  01_curve_overlays.png, 02_scatter_ellipses.png,
  03_correlation_panels.png, 04_qq_normality.png,
  05_tpc_prediction.png, summary.txt

Usage:
  python phasing_analysis.py            # full run, TPC as primary stratum
  python phasing_analysis.py --duration 36   # change example forecast duration

Requires: numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.special import betainc
from scipy.stats import chi2, pearsonr, shapiro, spearmanr

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG — edit here if your column names differ
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
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

COLORS = {"TPC": "#1f3864", "TEC": "#2e8b8b", "OPC": "#c9a227"}  # navy/teal/gold
FALLBACK_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ============================================================================
# STAGE 1 — LOAD & VALIDATE
# ============================================================================

def _resolve_columns(df: pd.DataFrame) -> dict:
    """Map logical names -> actual column names, honoring overrides."""
    norm = {c: c.strip().lower().replace(" ", "_") for c in df.columns}

    def find(candidates_exact, contains_all=None, contains_none=()):
        for actual, n in norm.items():
            if n in candidates_exact:
                return actual
        if contains_all:
            for actual, n in norm.items():
                if all(tok in n for tok in contains_all) and not any(
                    tok in n for tok in contains_none
                ):
                    return actual
        return None

    resolved = {
        "project": COL_OVERRIDES["project"]
        or find({"project", "project_id", "proj", "program"}),
        "cost_type": COL_OVERRIDES["cost_type"]
        or find({"cost_type", "costtype", "type", "cost_category"}),
        "pct_sched": COL_OVERRIDES["pct_sched"]
        or find(
            {"cum_pct_schedule", "cum_pct_sched", "pct_schedule", "pct_sched"},
            contains_all=("sched",),
        )
        or find(set(), contains_all=("time",), contains_none=("cost",)),
        "pct_cost": COL_OVERRIDES["pct_cost"]
        or find(
            {"cum_pct_cost", "pct_cost", "cum_cost_pct"},
            contains_all=("cost", "pct"),
            contains_none=("sched", "type"),
        )
        or find(set(), contains_all=("cost",), contains_none=("sched", "type")),
        "year": COL_OVERRIDES["year"]
        or find({"year", "fy", "fiscal_year", "yr"}),
    }

    # 'year' is optional — everything else is required
    missing = [k for k, v in resolved.items() if v is None and k != "year"]
    if missing:
        print(f"\nERROR: could not resolve columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        print("Set COL_OVERRIDES at the top of the script and re-run.")
        sys.exit(1)
    return resolved


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("STAGE 1: LOAD & VALIDATE")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Place data.csv next to this script.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    cols = _resolve_columns(df)
    print(f"Column mapping: {cols}")

    rename_map = {
        cols["project"]: "project",
        cols["cost_type"]: "cost_type",
        cols["pct_sched"]: "pct_sched",
        cols["pct_cost"]: "pct_cost",
    }
    if cols["year"]:
        rename_map[cols["year"]] = "year"
        print(f"  Year column detected: '{cols['year']}' "
              "(used for chronological sort + vintage drift check)")
    else:
        print("  No year column detected — sorting by pct_sched only")
    df = df.rename(columns=rename_map)
    df["cost_type"] = df["cost_type"].astype(str).str.strip().str.upper()

    # Coerce numerics; rescale 0-100 -> 0-1 if needed
    for c in ("pct_sched", "pct_cost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max() > 1.5:
            print(f"  {c}: values exceed 1.5 -> assuming percent, dividing by 100")
            df[c] = df[c] / 100.0

    n_bad = df[["pct_sched", "pct_cost"]].isna().any(axis=1).sum()
    if n_bad:
        print(f"  Dropping {n_bad} rows with non-numeric pct values")
        df = df.dropna(subset=["pct_sched", "pct_cost"])

    sort_cols = ["project", "cost_type"]
    if "year" in df.columns:
        sort_cols.append("year")   # true chronology first; robust to tied pct_sched
    sort_cols.append("pct_sched")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Monotonicity check (tolerate float noise of 1e-9)
    print("\nMonotonicity check:")
    n_viol = 0
    for (proj, ct), g in df.groupby(["project", "cost_type"]):
        if np.any(np.diff(g["pct_sched"].values) < -1e-9):
            print(f"  WARNING {proj}/{ct}: pct_sched not monotone")
            n_viol += 1
        if np.any(np.diff(g["pct_cost"].values) < -1e-9):
            print(f"  WARNING {proj}/{ct}: pct_cost not monotone")
            n_viol += 1
    n_curves = df.groupby(["project", "cost_type"]).ngroups
    print(f"  {n_curves} curves checked, {n_viol} violations"
          + (" (proceeding anyway)" if n_viol else ""))

    print("\nDataset summary:")
    print(f"  Projects:   {df['project'].nunique()}")
    print(f"  Cost types: {sorted(df['cost_type'].unique())}")
    pts = df.groupby(["project", "cost_type"]).size()
    print(f"  Points per curve: min={pts.min()}, median={pts.median():.0f}, "
          f"max={pts.max()}")

    return df, cols


# ============================================================================
# STAGE 2 — BETA CDF FITTING
# ============================================================================

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
    lo, hi = FIT_BOUNDS[0][0], FIT_BOUNDS[1][0]
    return float(np.clip(a, lo, hi)), float(np.clip(b, lo, hi))


def fit_all_curves(df: pd.DataFrame) -> pd.DataFrame:
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

        if len(t_fit) < MIN_INTERIOR_POINTS:
            print(f"  SKIP {proj}/{ct}: only {len(t_fit)} interior points")
            continue

        p0 = _moment_init(t_fit, y_fit)
        try:
            popt, pcov = curve_fit(
                beta_cdf, t_fit, y_fit, p0=p0, bounds=FIT_BOUNDS, maxfev=20000
            )
        except Exception as e:
            print(f"  FAIL {proj}/{ct}: {str(e)[:60]}")
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
            np.any(np.isclose(popt, FIT_BOUNDS[0], atol=1e-3))
            or np.any(np.isclose(popt, FIT_BOUNDS[1], atol=1e-3))
        )

        rows.append(
            dict(
                project=proj, cost_type=ct,
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
    if fits.empty:
        print("ERROR: no curves fitted. Check data.")
        sys.exit(1)

    print(f"\nFitted {len(fits)} curves")
    for ct, g in fits.groupby("cost_type"):
        print(f"  {ct}: n={len(g)}, median R2={g['r2'].median():.4f}, "
              f"mean RMSE={g['rmse'].mean():.4f}, at_bound={g['at_bound'].sum()}")
    nb = fits["at_bound"].sum()
    if nb:
        print(f"  NOTE: {nb} fits hit parameter bounds — inspect these curves.")
    return fits


# ============================================================================
# STAGE 3 — EDA
# ============================================================================

def _color_for(ct: str, i: int):
    return COLORS.get(ct, FALLBACK_COLORS[i % 10])


def _ordered_types(fits: pd.DataFrame) -> list[str]:
    present = list(fits["cost_type"].unique())
    ordered = [c for c in COST_TYPE_ORDER if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered


def plot_curve_overlays(df: pd.DataFrame, fits: pd.DataFrame):
    types = _ordered_types(fits)
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 5.5),
                             squeeze=False)
    t_plot = np.linspace(0, 1, 300)

    for i, ct in enumerate(types):
        ax = axes[0][i]
        sub = fits[fits["cost_type"] == ct]
        col = _color_for(ct, i)
        # raw data as faint dots
        raw = df[df["cost_type"] == ct]
        ax.plot(raw["pct_sched"], raw["pct_cost"], ".", ms=2, alpha=0.15,
                color="gray")
        for _, r in sub.iterrows():
            ax.plot(t_plot, beta_cdf(t_plot, r["alpha"], r["beta"]),
                    color=col, alpha=0.35, lw=1.2)
        # curve at the mean parameters
        ax.plot(t_plot, beta_cdf(t_plot, sub["alpha"].mean(), sub["beta"].mean()),
                color="black", lw=2.5, label="Curve at mean (ᾱ, β̄)")
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Linear phasing")
        ax.set_title(f"{ct}  (n={len(sub)} projects)", fontweight="bold")
        ax.set_xlabel("Cumulative % Schedule")
        ax.set_ylabel("Cumulative % Cost")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("Fitted Beta CDF Phasing Curves by Cost Type",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "01_curve_overlays.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _add_conf_ellipse(ax, x, y, color, q=0.95, **kw):
    if len(x) < 3:
        return
    data = np.column_stack([x, y])
    mean = data.mean(axis=0)
    cov = np.cov(data.T)
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    c = chi2.ppf(q, df=2)
    w, h = 2 * np.sqrt(c * evals[0]), 2 * np.sqrt(c * evals[1])
    ax.add_patch(Ellipse(mean, w, h, angle=angle, facecolor=color, alpha=0.12,
                         edgecolor=color, lw=2, ls="--", **kw))


def plot_scatter_ellipses(fits: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    types = _ordered_types(fits)

    for space, ax in zip(("unit", "log"), axes):
        for i, ct in enumerate(types):
            sub = fits[fits["cost_type"] == ct]
            col = _color_for(ct, i)
            x = sub["alpha"].values
            y = sub["beta"].values
            if space == "log":
                x, y = np.log(x), np.log(y)
            ax.scatter(x, y, s=90, color=col, edgecolor="black", lw=1,
                       alpha=0.8, label=f"{ct} (n={len(sub)})")
            _add_conf_ellipse(ax, x, y, col)
        if space == "unit":
            ax.set_xlabel("α"); ax.set_ylabel("β")
            ax.set_title("Unit space", fontweight="bold")
        else:
            ax.set_xlabel("ln α"); ax.set_ylabel("ln β")
            ax.set_title("Log space", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Fitted (α, β) with 95% Confidence Ellipses",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "02_scatter_ellipses.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# STAGE 4 — CORRELATION & NORMALITY
# ============================================================================

def correlation_analysis(fits: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STAGE 4: CORRELATION & NORMALITY")
    print("=" * 70)

    rows = []
    for ct, sub in fits.groupby("cost_type"):
        a, b = sub["alpha"].values, sub["beta"].values
        la, lb = np.log(a), np.log(b)
        r_u, p_u = pearsonr(a, b)
        r_l, p_l = pearsonr(la, lb)
        rho, p_rho = spearmanr(a, b)
        # Shapiro on marginals (log space) for BVN-in-log justification
        sw_la = shapiro(la).pvalue if len(la) >= 3 else np.nan
        sw_lb = shapiro(lb).pvalue if len(lb) >= 3 else np.nan
        sw_a = shapiro(a).pvalue if len(a) >= 3 else np.nan
        sw_b = shapiro(b).pvalue if len(b) >= 3 else np.nan
        rows.append(dict(
            cost_type=ct, n=len(sub),
            pearson_unit=r_u, p_unit=p_u,
            pearson_log=r_l, p_log=p_l,
            spearman=rho, p_spearman=p_rho,
            shapiro_p_alpha=sw_a, shapiro_p_beta=sw_b,
            shapiro_p_log_alpha=sw_la, shapiro_p_log_beta=sw_lb,
        ))
        print(f"\n{ct} (n={len(sub)}):")
        print(f"  Pearson r  unit: {r_u:+.3f} (p={p_u:.3f})   "
              f"log: {r_l:+.3f} (p={p_l:.3f})")
        print(f"  Spearman ρ:      {rho:+.3f} (p={p_rho:.3f})")
        print(f"  Shapiro p (α, β) unit: ({sw_a:.3f}, {sw_b:.3f})   "
              f"log: ({sw_la:.3f}, {sw_lb:.3f})")

    corr = pd.DataFrame(rows)
    corr.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    return corr


def plot_correlation_panels(fits: pd.DataFrame):
    types = _ordered_types(fits)
    fig, axes = plt.subplots(len(types), 2, figsize=(12, 4.5 * len(types)),
                             squeeze=False)
    for i, ct in enumerate(types):
        sub = fits[fits["cost_type"] == ct]
        col = _color_for(ct, i)
        for j, space in enumerate(("unit", "log")):
            ax = axes[i][j]
            x = sub["alpha"].values
            y = sub["beta"].values
            if space == "log":
                x, y = np.log(x), np.log(y)
            ax.scatter(x, y, s=70, color=col, edgecolor="black", alpha=0.8)
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, np.polyval(z, xs), "r--", lw=2)
                r, p = pearsonr(x, y)
                ax.set_title(f"{ct} — {space} space   r={r:+.3f} (p={p:.3f})",
                             fontweight="bold", fontsize=11)
            lbl = ("α", "β") if space == "unit" else ("ln α", "ln β")
            ax.set_xlabel(lbl[0]); ax.set_ylabel(lbl[1])
            ax.grid(alpha=0.3)
    fig.suptitle("α–β Correlation by Cost Type", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "03_correlation_panels.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_qq_normality(fits: pd.DataFrame):
    """Mahalanobis distance^2 vs chi2(2) quantiles, unit vs log space."""
    types = _ordered_types(fits)
    fig, axes = plt.subplots(len(types), 2, figsize=(12, 4.2 * len(types)),
                             squeeze=False)
    for i, ct in enumerate(types):
        sub = fits[fits["cost_type"] == ct]
        for j, space in enumerate(("unit", "log")):
            ax = axes[i][j]
            data = sub[["alpha", "beta"]].values
            if space == "log":
                data = np.log(data)
            if len(data) < 4:
                ax.set_visible(False)
                continue
            mean = data.mean(axis=0)
            cov = np.cov(data.T)
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                ax.set_visible(False)
                continue
            d2 = np.sort(np.einsum("ij,jk,ik->i", data - mean, inv, data - mean))
            n = len(d2)
            q = chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=2)
            ax.scatter(q, d2, s=60, edgecolor="black", alpha=0.8)
            m = max(q.max(), d2.max())
            ax.plot([0, m], [0, m], "r--", lw=1.5)
            ax.set_title(f"{ct} — {space} space", fontweight="bold", fontsize=11)
            ax.set_xlabel("χ²(2) quantiles")
            ax.set_ylabel("Mahalanobis d²")
            ax.grid(alpha=0.3)
    fig.suptitle("Bivariate Normality Check (points on line ⇒ BVN plausible)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "04_qq_normality.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# STAGE 5 — JOINT DISTRIBUTIONS + PREDICTOR
# ============================================================================

def fit_joint_distributions(fits: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("STAGE 5: JOINT DISTRIBUTION FITS")
    print("=" * 70)

    dists = {"bvn_unit": [], "bvn_log": []}
    for ct, sub in fits.groupby("cost_type"):
        data_u = sub[["alpha", "beta"]].values
        data_l = np.log(data_u)
        for key, d in (("bvn_unit", data_u), ("bvn_log", data_l)):
            mean = d.mean(axis=0)
            cov = np.cov(d.T)
            sd = np.sqrt(np.diag(cov))
            rho = cov[0, 1] / (sd[0] * sd[1]) if sd.prod() > 0 else 0.0
            dists[key].append(dict(
                cost_type=ct, n=int(len(sub)),
                mu=[float(mean[0]), float(mean[1])],
                cov=[[float(cov[0, 0]), float(cov[0, 1])],
                     [float(cov[1, 0]), float(cov[1, 1])]],
                sigma=[float(sd[0]), float(sd[1])],
                rho=float(rho),
            ))
        u = dists["bvn_unit"][-1]
        l = dists["bvn_log"][-1]
        print(f"\n{ct}:")
        print(f"  Unit BVN: μ=({u['mu'][0]:.3f},{u['mu'][1]:.3f}) "
              f"σ=({u['sigma'][0]:.3f},{u['sigma'][1]:.3f}) ρ={u['rho']:+.3f}")
        print(f"  Log  BVN: μ=({l['mu'][0]:.3f},{l['mu'][1]:.3f}) "
              f"σ=({l['sigma'][0]:.3f},{l['sigma'][1]:.3f}) ρ={l['rho']:+.3f}")

    with open(OUTPUT_DIR / "distributions.json", "w") as f:
        json.dump(dists, f, indent=2)
    print(f"\nSaved -> {OUTPUT_DIR / 'distributions.json'}")
    return dists


class PhasingPredictor:
    """
    Samples (α, β) for a future project and evaluates Beta CDF phasing.

    Default model 'bvn_log': (ln α, ln β) ~ BVN  (bivariate lognormal).
    Guarantees positive parameters; captures correlation; right-skew friendly.
    'bvn_unit' samples in unit space with rejection for positivity.
    """

    def __init__(self, dists: dict, model: str = "bvn_log",
                 rng: np.random.Generator | None = None):
        self.model = model
        self.params = {p["cost_type"]: p for p in dists[model]}
        self.rng = rng or np.random.default_rng(RNG_SEED)

    def sample(self, cost_type: str, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        p = self.params[cost_type]
        mu, cov = np.array(p["mu"]), np.array(p["cov"])
        if self.model == "bvn_log":
            z = self.rng.multivariate_normal(mu, cov, size=n)
            ab = np.exp(z)
        else:  # bvn_unit with rejection for positivity
            out = []
            need = n
            while need > 0:
                z = self.rng.multivariate_normal(mu, cov, size=max(need * 2, 64))
                z = z[(z > 0.05).all(axis=1)]
                out.append(z[:need])
                need = n - sum(len(o) for o in out)
            ab = np.vstack(out)
        return ab[:, 0], ab[:, 1]

    def predict(self, cost_type: str, duration_months: int,
                n_samples: int = N_PRED_SAMPLES):
        """Monthly cumulative phasing samples over a given duration.
        Returns (months 0..D, samples of shape (n_samples, D+1))."""
        months = np.arange(0, duration_months + 1)
        t = months / duration_months
        a, b = self.sample(cost_type, n_samples)
        cum = beta_cdf(t[None, :], a[:, None], b[:, None])
        cum[:, 0], cum[:, -1] = 0.0, 1.0
        return months, cum

    def quantile_table(self, cost_type: str, duration_months: int,
                       qs=(0.05, 0.25, 0.50, 0.75, 0.95),
                       n_samples: int = N_PRED_SAMPLES) -> pd.DataFrame:
        months, cum = self.predict(cost_type, duration_months, n_samples)
        out = {"month": months}
        for q in qs:
            out[f"q{int(q * 100):02d}"] = np.quantile(cum, q, axis=0)
        out["mean"] = cum.mean(axis=0)
        return pd.DataFrame(out)


def plot_prediction(pred: PhasingPredictor, cost_type: str,
                    duration_months: int):
    months, cum = pred.predict(cost_type, duration_months)
    q05, q25, q50, q75, q95 = (np.quantile(cum, q, axis=0)
                               for q in (0.05, 0.25, 0.50, 0.75, 0.95))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: cumulative fan chart
    ax = axes[0]
    col = COLORS.get(cost_type, "#1f3864")
    ax.fill_between(months, q05, q95, alpha=0.2, color=col, label="90% interval")
    ax.fill_between(months, q25, q75, alpha=0.35, color=col, label="50% interval")
    ax.plot(months, q50, color=col, lw=2.5, label="Median")
    # a few spaghetti draws for texture
    for k in range(12):
        ax.plot(months, cum[k], color="gray", alpha=0.3, lw=0.8)
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative % of Total Cost")
    ax.set_title(f"{cost_type} Cumulative Phasing — {duration_months}-month project",
                 fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xlim(0, duration_months); ax.set_ylim(0, 1.02)

    # Panel 2: incremental (monthly) spend distribution
    ax = axes[1]
    inc = np.diff(cum, axis=1)
    i05, i50, i95 = (np.quantile(inc, q, axis=0) for q in (0.05, 0.50, 0.95))
    mid = months[1:]
    ax.fill_between(mid, i05, i95, alpha=0.2, color=col, label="90% interval")
    ax.plot(mid, i50, color=col, lw=2.5, label="Median")
    ax.set_xlabel("Month"); ax.set_ylabel("Monthly % of Total Cost")
    ax.set_title("Implied Monthly Spend Profile", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, duration_months)

    fig.suptitle(f"Stochastic Phasing Forecast ({pred.model})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "05_tpc_prediction.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# STAGE 6 — SUMMARY
# ============================================================================

def write_summary(fits: pd.DataFrame, corr: pd.DataFrame, dists: dict,
                  duration: int):
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
    w("FITS:")
    for ct, g in fits.groupby("cost_type"):
        w(f"  {ct}: n={len(g)}  median R2={g['r2'].median():.4f}  "
          f"mean RMSE={g['rmse'].mean():.4f}")
        w(f"       α: mean={g['alpha'].mean():.3f} sd={g['alpha'].std():.3f}   "
          f"β: mean={g['beta'].mean():.3f} sd={g['beta'].std():.3f}")
        w(f"       mean spend timing (α/(α+β)): "
          f"{g['mean_timing'].mean():.3f}")
    w("")
    w("CORRELATION (α vs β):")
    w(corr.to_string(index=False))
    if "vintage" in fits.columns and fits["vintage"].notna().any():
        w("")
        w("TEMPORAL DRIFT CHECK (parameters vs project start year, Spearman):")
        for ct, g in fits.groupby("cost_type"):
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
    w("")
    w("RECOMMENDED PREDICTIVE MODEL: bvn_log (bivariate lognormal)")
    w("  Sample (ln α, ln β) ~ BVN(μ, Σ); exponentiate; evaluate Beta CDF at")
    w("  t = month/duration for any stochastic duration draw.")
    w("")
    for p in dists["bvn_log"]:
        w(f"  {p['cost_type']}: μ_ln=({p['mu'][0]:.4f}, {p['mu'][1]:.4f})  "
          f"σ_ln=({p['sigma'][0]:.4f}, {p['sigma'][1]:.4f})  ρ={p['rho']:+.4f}")
    w("")
    w(f"Example forecast generated for {PRIMARY_COST_TYPE}, "
      f"{duration}-month duration (05_tpc_prediction.png).")
    w("")
    w("CAVEATS:")
    w("  - n=18 projects per stratum: parameter-distribution estimates carry")
    w("    sampling uncertainty not reflected in the fitted BVN (a hierarchical")
    w("    Bayes extension would propagate it).")
    w("  - Fits assume the Beta CDF family is adequate; check curve overlays")
    w("    and max_abs_err in fits.csv for systematic misfit.")

    text = "\n".join(lines)
    print(OUTPUT_DIR)
    (OUTPUT_DIR / "summary.txt").write_text(text)
    print("\n" + text)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=24,
                    help="Example forecast duration in months (default 24)")
    ap.add_argument("--model", choices=["bvn_log", "bvn_unit"],
                    default="bvn_log", help="Predictor sampling model")
    args = ap.parse_args()

    np.random.seed(RNG_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, _ = load_and_validate()
    fits = fit_all_curves(df)
    fits.to_csv(OUTPUT_DIR / "fits.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'fits.csv'}")

    print("\n" + "=" * 70)
    print("STAGE 3: EDA PLOTS")
    print("=" * 70)
    plot_curve_overlays(df, fits)
    plot_scatter_ellipses(fits)
    print("Saved -> 01_curve_overlays.png, 02_scatter_ellipses.png")

    corr = correlation_analysis(fits)
    plot_correlation_panels(fits)
    plot_qq_normality(fits)
    print("Saved -> 03_correlation_panels.png, 04_qq_normality.png")

    dists = fit_joint_distributions(fits)

    if PRIMARY_COST_TYPE in fits["cost_type"].values:
        pred = PhasingPredictor(dists, model=args.model)
        plot_prediction(pred, PRIMARY_COST_TYPE, args.duration)
        qt = pred.quantile_table(PRIMARY_COST_TYPE, args.duration)
        qt.to_csv(OUTPUT_DIR / "tpc_quantile_table.csv", index=False)
        print(f"Saved -> 05_tpc_prediction.png, tpc_quantile_table.csv")
    else:
        print(f"WARNING: {PRIMARY_COST_TYPE} not in data; skipping prediction.")

    write_summary(fits, corr, dists, args.duration)
    print("\nDONE. All outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
