"""
Phasing deck figure generation
==============================

Regenerates every figure in the stochastic-phasing presentation from real fitted
data produced by ``phasing_analysis.py``:

    fits.csv per-curve fitted (alpha, beta), one row per project x cost type
    distributions.json fitted population parameters (bvn_log preferred)

If either input is missing, the module falls back to a documented illustrative
population and stamps every figure it produces with an ILLUSTRATIVE watermark, so
placeholder figures cannot silently end up in a briefing package.

Usage
-----
    # easiest: as part of a pipeline run, straight from that run's fits
    python phasing_analysis.py --data path/to/data.csv --presentation

    # standalone, from a previous run's output files; TPC stratum
    python -m phasing.presentation_figures --fits outputs/fits.csv \
        --dists outputs/distributions.json --cost-type TPC --out presentation_figures/

    # tune the companion cost/schedule models used by the Monte Carlo figures
    python -m phasing.presentation_figures --fits outputs/fits.csv \
        --dists outputs/distributions.json \
        --duration-median 62 --duration-sigma 0.30 \
        --cost-median 420 --cost-sigma 0.18 --n-fy 10

    # illustrative mode (no inputs) -- every figure is watermarked
    python -m phasing.presentation_figures --out presentation_figures/

    # single figure while iterating
    python -m phasing.presentation_figures --fits ... --dists ... --only f07

Figures
-------
    f01 historical curve fan (normalized), spread annotated at 50% schedule
    f02 Beta shape atlas as density/CDF pairs
    f03 fitted (alpha, beta) scatter with 95% ellipses, unit and log space
    f03a same, in the mean-precision (mu, nu) parameterization
    f04 where the uncertainty lives: normalized time vs calendar time
    f05 a late fiscal year as a mixture distribution
    f06 unconditional annual quantiles, showing the P50 truncation
    f07 conditional annual spend given active, with P(active)
    f08 cumulative fan chart with completion markers
    f09 the phasing estimating relationship: formulas + strip areas

Requires: numpy, pandas, scipy, matplotlib. No other dependencies.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.special import logit
from scipy.stats import beta as sbeta, chi2

from .fitting import beta_cdf as bcdf
from .parameterization import ab_to_mu_nu
from .style import conf_ellipse

# ============================================================================
# Palette -- matches the CECOP dark-theme template
# ============================================================================
NAVY = "#021E42"
DEEP = "#14233A"
SLATE = "#42566B"
TEAL = "#0D9488"
DTEAL = "#0A7A70"
LTEAL = "#5EEAD4"
GOLD = "#B8860B"
LGOLD = "#D4B36A"
CRIM = "#C50E3D"
MINT = "#ECFDF5"
CREAM = "#FBF7F2"
ICE = "#F7FAFC"
BORD = "#D9E2EC"

COST_TYPE_COLOR = {"TPC": NAVY, "TEC": TEAL, "OPC": GOLD}
FALLBACK_CYCLE = [TEAL, GOLD, NAVY, CRIM, DTEAL, SLATE]

def apply_style() -> None:
    """Set global matplotlib defaults for the deck's visual language."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "mathtext.fontset": "dejavusans",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": DEEP,
            "text.color": DEEP,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "axes.titlecolor": DEEP,
            "axes.grid": True,
            "grid.color": BORD,
            "grid.alpha": 0.9,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

# ============================================================================
# Configuration
# ============================================================================

#: Illustrative population, used only when no fitted inputs are supplied.
#: These are NOT program values. They exist so the deck can be built end to end
#: before the real fits are available.
ILLUSTRATIVE_MU_LN = np.array([np.log(2.15), np.log(1.90)])
ILLUSTRATIVE_SIGMA_LN = np.array([0.26, 0.20])
ILLUSTRATIVE_RHO_LN = 0.55
ILLUSTRATIVE_N_PROJECTS = 16

@dataclass
class SimConfig:
    """Companion cost/schedule models and fiscal-year binning for figures 4-8.

    The phasing model itself is fitted; these are the *other* two marginals that
    phasing has to be coupled with. Override them from the CLI to match the
    program's actual cost and schedule risk analyses.
    """

    duration_median: float = 62.0 # months
    duration_sigma: float = 0.30 # lognormal sigma
    duration_clip: tuple[float, float] = (24.0, 150.0)
    cost_median: float = 420.0 # $M
    cost_sigma: float = 0.18
    cost_clip: tuple[float, float] = (100.0, 1200.0)
    n_iter: int = 20_000
    max_month: int = 144
    fy_months: int = 12
    n_fy: int = 10
    active_threshold: float = 0.5 # $M below which a FY counts as inactive
    mixture_fy: int = 8 # 1-indexed FY shown in f05
    calendar_month: int = 54 # month annotated in f04 right panel
    seed: int = 11

@dataclass
class Inputs:
    """Fitted inputs for the figures, or an illustrative stand-in."""

    alpha: np.ndarray # per-curve fitted alpha (selected stratum)
    beta: np.ndarray # per-curve fitted beta
    mu_ln: np.ndarray # population mean of (ln alpha, ln beta)
    cov_ln: np.ndarray # population covariance in log space
    cost_type: str # stratum label, e.g. "TPC"
    illustrative: bool # stamp figures if True
    all_fits: pd.DataFrame | None = None # every stratum, for f03
    notes: list[str] = field(default_factory=list)

    def draw(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample n (alpha, beta) pairs from the fitted log-space population."""
        z = rng.multivariate_normal(self.mu_ln, self.cov_ln, size=n)
        ab = np.exp(z)
        return ab[:, 0], ab[:, 1]

# ============================================================================
# Input loading
# ============================================================================

def _illustrative_inputs(rng: np.random.Generator, cost_type: str) -> Inputs:
    s, r = ILLUSTRATIVE_SIGMA_LN, ILLUSTRATIVE_RHO_LN
    cov = np.array([[s[0] ** 2, r * s[0] * s[1]], [r * s[0] * s[1], s[1] ** 2]])
    ab = np.exp(rng.multivariate_normal(ILLUSTRATIVE_MU_LN, cov,
                                        size=ILLUSTRATIVE_N_PROJECTS))
    df = pd.DataFrame({"alpha": ab[:, 0], "beta": ab[:, 1], "cost_type": cost_type})
    return Inputs(
        alpha=ab[:, 0], beta=ab[:, 1], mu_ln=ILLUSTRATIVE_MU_LN, cov_ln=cov,
        cost_type=cost_type, illustrative=True, all_fits=df,
        notes=["no fits.csv / distributions.json supplied -- illustrative population"],
    )

def build_inputs(fits: pd.DataFrame, dists: dict | None, cost_type: str,
                 notes: list[str] | None = None) -> Inputs:
    """Build figure inputs from in-memory pipeline results.

    Parameters
    ----------
    fits : pd.DataFrame
        Fitted curves with positive alpha/beta and a cost_type column. The
        pipeline passes its Stage-2-screened fits (bound-contact excluded);
        `load_inputs` applies the equivalent cleanup to a raw fits.csv first.
    dists : dict or None
        Output of `distributions.fit_joint_distributions` (or the parsed
        distributions.json). Only the ``bvn_log`` entry is usable here --
        ``bvn_unit``/``bvn_munu`` parameters live in other coordinate systems
        and cannot be exponentiated into (alpha, beta). If no ``bvn_log``
        entry matches, the population is estimated from the fitted curves.
    cost_type : str
        Stratum to feature (e.g. "TPC").
    """
    notes = list(notes or [])
    fits = fits.copy()
    fits["cost_type"] = fits["cost_type"].astype(str).str.strip().str.upper()
    sub = fits[fits["cost_type"] == cost_type.upper()]
    if sub.empty:
        raise SystemExit(
            f"No fitted curves for cost_type={cost_type!r}. "
            f"Available: {sorted(fits['cost_type'].unique())}"
        )

    alpha = sub["alpha"].to_numpy(float)
    beta = sub["beta"].to_numpy(float)

    mu_ln = cov_ln = None
    for entry in (dists or {}).get("bvn_log", []):
        if str(entry.get("cost_type", "")).upper() == cost_type.upper():
            mu_ln = np.asarray(entry["mu"], float)
            cov_ln = np.asarray(entry["cov"], float)
            break

    if mu_ln is None:
        d = np.column_stack([np.log(alpha), np.log(beta)])
        mu_ln, cov_ln = d.mean(0), np.cov(d.T)
        notes.append("population estimated from fitted curves "
                     "(no usable bvn_log distribution entry)")

    return Inputs(alpha=alpha, beta=beta, mu_ln=mu_ln, cov_ln=cov_ln,
                  cost_type=cost_type.upper(), illustrative=False,
                  all_fits=fits, notes=notes)


def load_inputs(fits_path: str | None, dists_path: str | None, cost_type: str,
                rng: np.random.Generator) -> Inputs:
    """Load fitted curves and population parameters, falling back to illustrative.

    Parameters
    ----------
    fits_path : str or None
        Path to ``fits.csv`` written by ``phasing_analysis.py``.
    dists_path : str or None
        Path to ``distributions.json``. The ``bvn_log`` entry is preferred; if
        absent, the population is estimated directly from the fitted curves.
    cost_type : str
        Stratum to feature (e.g. "TPC").
    """
    if not fits_path:
        return _illustrative_inputs(rng, cost_type)

    fits = pd.read_csv(fits_path)
    if "cost_type" not in fits.columns:
        fits["cost_type"] = cost_type

    notes: list[str] = []

    # Drop curves that cannot inform the population: failed fits, bound contact.
    n0 = len(fits)
    if "at_bound" in fits.columns:
        pinned = fits["at_bound"].astype(str).str.upper().isin(["TRUE", "1"])
        if pinned.any():
            notes.append(f"excluded {int(pinned.sum())} bound-contact curve(s) "
                         "-- weakly identified, would inflate population spread")
            fits = fits[~pinned]
    fits = fits.dropna(subset=["alpha", "beta"])
    fits = fits[(fits["alpha"] > 0) & (fits["beta"] > 0)]
    if len(fits) < n0:
        notes.append(f"using {len(fits)} of {n0} fitted curves")

    dists = None
    if dists_path and Path(dists_path).exists():
        dists = json.loads(Path(dists_path).read_text())

    return build_inputs(fits, dists, cost_type, notes=notes)

# ============================================================================
# Simulation shared by figures 4-8
# ============================================================================

@dataclass
class SimResult:
    months: np.ndarray # 0 .. max_month
    cum: np.ndarray # (n_iter, n_months) cumulative $ spend
    pct_samp: np.ndarray # (n_iter, n_months) cumulative fraction, shape sampled
    pct_fixed: np.ndarray # (n_iter, n_months) cumulative fraction, shape at mean
    shape_only: np.ndarray # (n_iter, 101) cumulative fraction on normalized time
    tn_grid: np.ndarray # normalized-time grid for shape_only
    fy: np.ndarray # (n_iter, n_fy) $ spend per fiscal year
    active: np.ndarray # (n_iter, n_fy) bool
    p_active: np.ndarray # (n_fy,)
    duration: np.ndarray
    cost: np.ndarray
    fy_labels: list[str]

def simulate(inp: Inputs, cfg: SimConfig, rng: np.random.Generator) -> SimResult:
    """Couple cost, duration, and phasing shape into fiscal-year spend streams."""
    n = cfg.n_iter
    D = np.clip(rng.lognormal(np.log(cfg.duration_median), cfg.duration_sigma, n),
                *cfg.duration_clip)
    C = np.clip(rng.lognormal(np.log(cfg.cost_median), cfg.cost_sigma, n),
                *cfg.cost_clip)
    A, B = inp.draw(rng, n)

    months = np.arange(cfg.max_month + 1)
    tn = np.clip(months[None, :] / D[:, None], 0.0, 1.0)

    pct_samp = bcdf(tn, A[:, None], B[:, None])
    a0, b0 = np.exp(inp.mu_ln)
    pct_fixed = bcdf(tn, a0, b0)
    cum = pct_samp * C[:, None]

    tn_grid = np.linspace(0.0, 1.0, 101)
    shape_only = bcdf(tn_grid[None, :], A[:, None], B[:, None])

    inc = np.diff(cum, axis=1, prepend=0.0)
    fy = np.array([inc[:, y * cfg.fy_months:(y + 1) * cfg.fy_months].sum(1)
                   for y in range(cfg.n_fy)]).T
    active = fy > cfg.active_threshold

    return SimResult(
        months=months, cum=cum, pct_samp=pct_samp, pct_fixed=pct_fixed,
        shape_only=shape_only, tn_grid=tn_grid, fy=fy, active=active,
        p_active=active.mean(0), duration=D, cost=C,
        fy_labels=[f"FY{i + 1}" for i in range(cfg.n_fy)],
    )

# ============================================================================
# Plot helpers
# ============================================================================

def _stamp(fig, inp: Inputs) -> None:
    """Watermark illustrative figures so they cannot be mistaken for fitted ones."""
    if not inp.illustrative:
        return
    fig.text(0.995, 0.008, "ILLUSTRATIVE \u2014 not fitted to program data",
             ha="right", va="bottom", fontsize=8.5, color=CRIM,
             fontweight="bold", alpha=0.85)

def _save(fig, out: Path, name: str, inp: Inputs, tight: bool = False) -> Path:
    _stamp(fig, inp)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print(f" wrote {path}")
    return path

def _color_for(cost_type: str, i: int = 0) -> str:
    return COST_TYPE_COLOR.get(cost_type.upper(), FALLBACK_CYCLE[i % len(FALLBACK_CYCLE)])

def _ellipse_coverage(x: np.ndarray, y: np.ndarray, q: float = 0.95) -> float | None:
    """Empirical fraction of (x, y) inside their own q-level BVN ellipse.

    Goodness-of-fit for the elliptical/BVN approximation drawn by
    `conf_ellipse`: under a true bivariate normal this should land near `q`.
    Mirrors the mean/cov/Mahalanobis math in `style.conf_ellipse`.
    """
    if len(x) < 3:
        return None
    d = np.column_stack([x, y])
    mean, cov = d.mean(axis=0), np.cov(d.T)
    diff = d - mean
    # pinv, not inv: near-singular cov (collinear or very tight clusters)
    # would otherwise raise LinAlgError -- conf_ellipse sidesteps the same
    # issue via eigh, this mirrors that robustness.
    m2 = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(cov), diff)
    return float(np.mean(m2 <= chi2.ppf(q, df=2)))


# ============================================================================
# Figures
# ============================================================================

def fig01_curve_fan(inp: Inputs, cfg: SimConfig, out: Path) -> None:
    """Fitted curves for the featured stratum, with the spread at mid-schedule."""
    t = np.linspace(0, 1, 300)
    a, b = inp.alpha, inp.beta
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for ai, bi in zip(a, b):
        ax.plot(t, bcdf(t, ai, bi), color=TEAL, alpha=0.55, lw=1.6)
    a0, b0 = np.exp(inp.mu_ln)
    ax.plot(t, bcdf(t, a0, b0), color=NAVY, lw=3, label="Curve at population mean")
    ax.plot([0, 1], [0, 1], ls=":", color=SLATE, lw=1.4, label="Linear phasing")

    ax.axvline(0.5, color=GOLD, lw=1.4, ls="--")
    mid = bcdf(0.5, a, b)
    lo, hi = mid.min(), mid.max()
    ax.annotate(f"At 50% schedule:\nspend ranged {lo * 100:.0f}%\u2013{hi * 100:.0f}%",
                xy=(0.5, (lo + hi) / 2), xytext=(0.60, 0.28), fontsize=11,
                color=GOLD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    ax.set_xlabel("Cumulative % of schedule")
    ax.set_ylabel("Cumulative % of cost")
    ax.set_title(f"{inp.cost_type} \u2014 {len(a)} completed projects",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "f01_curve_fan", inp)

def fig02_beta_atlas(inp: Inputs, cfg: SimConfig, out: Path) -> None:
    """Canonical Beta shapes as density/CDF pairs -- density on top, CDF below."""
    cases = [((1.4, 3.0), "Front-loaded", "\u03b1 < \u03b2", TEAL),
             ((3.0, 1.4), "Back-loaded", "\u03b1 > \u03b2", GOLD),
             ((3.0, 3.0), "Symmetric S", "\u03b1 = \u03b2 > 1", NAVY),
             ((1.0, 1.0), "Level / linear", "\u03b1 = \u03b2 = 1", SLATE)]
    t = np.linspace(1e-4, 1 - 1e-4, 400)
    fig = plt.figure(figsize=(13.2, 4.4))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.25, 1.0], hspace=0.34,
                  wspace=0.26, left=0.075, right=0.985, top=0.83, bottom=0.13)
    for k, ((A, B), ttl, sub, c) in enumerate(cases):
        mu, nu = A / (A + B), A + B
        pdf, cdf = sbeta.pdf(t, A, B), bcdf(t, A, B)

        ax = fig.add_subplot(gs[0, k])
        ax.plot(t, pdf, color=c, lw=2.4)
        ax.fill_between(t, 0, pdf, color=c, alpha=0.16)
        ax.axvline(mu, color=CRIM, lw=1.5, ls="--")
        ax.annotate(f"\u03bc={mu:.2f}", xy=(mu, ax.get_ylim()[1] * 0.97),
                    xytext=(4, 6), textcoords="offset points", color=CRIM,
                    fontweight="bold", fontsize=9)
        if A > 1 and B > 1:
            mode = (A - 1) / (A + B - 2)
            ax.plot([mode], [sbeta.pdf(mode, A, B)], marker="v", ms=7,
                    color=DEEP, zorder=5)
        ax.set_title(f"{ttl}\n{sub} \u03bd={nu:.1f}", fontsize=10,
                     fontweight="bold", color=c, linespacing=1.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, max(2.6, pdf.max() * 1.22))
        ax.tick_params(labelleft=False)
        if k == 0:
            ax.set_ylabel("spend rate\nf(t)", fontsize=9.5, color=DEEP, linespacing=1.4)

        ax = fig.add_subplot(gs[1, k])
        ax.plot(t, cdf, color=c, lw=2.4)
        ax.plot([0, 1], [0, 1], ls=":", color=BORD, lw=1.3)
        ax.axvline(mu, color=CRIM, lw=1.2, ls="--", alpha=0.75)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_xlabel("% of schedule", fontsize=9)
        if k == 0:
            ax.set_ylabel("cumulative\nF(t)", fontsize=9.5, color=DEEP, linespacing=1.4)

    fig.text(0.5, 0.955,
             "Read the top row: the density is the spend RATE \u2014 where the dollars "
             "actually fall. Bottom row: the same four shapes as cumulative curves.",
             ha="center", fontsize=10.5, color=DEEP, style="italic")
    # Canonical shapes, not fitted values -- no watermark needed.
    fig.savefig(out / "f02_beta_atlas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" wrote {out / 'f02_beta_atlas.png'}")

def _fig03_core(inp: Inputs, out: Path, name: str, panels: list[tuple]) -> None:
    """Shared scatter + 95% ellipse + coverage-GoF panel pair for f03 / f03a.

    panels: list of (title, xy_fn, xlabel, ylabel, ref), where xy_fn maps
    (alpha_array, beta_array) -> (x_array, y_array) for that panel, and ref
    is either "diag" (draw the x=y reference line -- meaningful when the axes
    ARE alpha/beta) or a float (draw a vertical reference line there instead,
    e.g. at mu=0.5 for the symmetric-curve reference in mu/nu space).
    """
    df = inp.all_fits if inp.all_fits is not None else pd.DataFrame(
        {"alpha": inp.alpha, "beta": inp.beta, "cost_type": inp.cost_type})
    types = [t for t in ["TPC", "TEC", "OPC"] if t in set(df["cost_type"])]
    types += [t for t in sorted(set(df["cost_type"])) if t not in types]

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, (title, xy_fn, xlab, ylab, ref) in zip(axs, panels):
        cov_lines = []
        for i, ct in enumerate(types):
            sub = df[df["cost_type"] == ct]
            X, Y = xy_fn(sub["alpha"].to_numpy(float), sub["beta"].to_numpy(float))
            c = _color_for(ct, i)
            ax.scatter(X, Y, s=85, color=c, edgecolor=NAVY, lw=1.1, zorder=3,
                       alpha=0.9, label=f"{ct} (n={len(sub)})")
            conf_ellipse(ax, X, Y, c)
            cov = _ellipse_coverage(X, Y)
            if cov is not None:
                cov_lines.append((c, f"{ct} {cov:.0%} (n={len(sub)})"))
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ref == "diag":
            lim = ax.get_xlim()
            ax.plot(lim, lim, ls=":", color=BORD, lw=1.2)
            ax.set_xlim(lim)
        else:
            ax.axvline(ref, ls=":", color=BORD, lw=1.2)
        if cov_lines:
            ax.text(0.97, 0.95, "95% ellipse coverage", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color=DEEP, fontweight="bold")
            for j, (c, txt) in enumerate(cov_lines):
                ax.text(0.97, 0.885 - 0.065 * j, txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=8.5, color=c)
    axs[0].text(0.03, 0.05, "each point = one completed project",
                transform=axs[0].transAxes, fontsize=9.5, color=SLATE,
                va="bottom", style="italic")
    axs[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, out, name, inp)

def fig03_scatter_ellipse(inp: Inputs, cfg: SimConfig, out: Path) -> None:
    """Fitted (alpha, beta) by stratum with 95% ellipses, unit and log space."""
    _fig03_core(inp, out, "f03_scatter_ellipse", [
        ("Unit space", lambda a, b: (a, b), "\u03b1", "\u03b2", "diag"),
        ("Log space", lambda a, b: (np.log(a), np.log(b)), "ln \u03b1", "ln \u03b2", "diag"),
    ])

def fig03a_scatter_ellipse_munu(inp: Inputs, cfg: SimConfig, out: Path) -> None:
    """Same fitted curves in the mean-precision (mu, nu) parameterization.

    mu = alpha / (alpha + beta) (mean spend timing), nu = alpha + beta
    (precision/concentration) -- see parameterization.ab_to_mu_nu. The log
    panel uses (logit mu, log nu), the exact coordinates distributions.py
    fits as "bvn_munu".
    """
    def unit_xy(a, b):
        return ab_to_mu_nu(a, b)

    def log_xy(a, b):
        mu, nu = ab_to_mu_nu(a, b)
        return logit(mu), np.log(nu)

    _fig03_core(inp, out, "f03a_scatter_ellipse_munu", [
        ("(\u03bc, \u03bd) space", unit_xy, "\u03bc", "\u03bd", 0.5),
        ("(logit \u03bc, log \u03bd) space", log_xy, "logit \u03bc", "log \u03bd", 0.0),
    ])

def fig04_coupling(inp: Inputs, cfg: SimConfig, out: Path, sim: SimResult) -> None:
    """Where each uncertainty lives: normalized time vs calendar time."""
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.6))
    tn = sim.tn_grid

    ax = axs[0]
    q5, q50, q95 = (np.percentile(sim.shape_only, q, axis=0) for q in (5, 50, 95))
    ax.fill_between(tn * 100, q5 * 100, q95 * 100, color=TEAL, alpha=0.25,
                    label="90% interval (shape)")
    ax.plot(tn * 100, q50 * 100, color=NAVY, lw=2.6, label="Median")
    a0, b0 = np.exp(inp.mu_ln)
    ax.plot(tn * 100, bcdf(tn, a0, b0) * 100, color=CRIM, lw=2.2, ls="--",
            label="Deterministic template")
    i = 50
    w = (q95[i] - q5[i]) * 100
    ax.annotate(f"at 50% schedule:\n{q5[i] * 100:.0f}\u2013{q95[i] * 100:.0f}% spent\n"
                f"({w:.0f}-point band)", xy=(50, q50[i] * 100), xytext=(8, 68),
                fontsize=10.5, color=GOLD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    ax.set_xlabel("% of schedule elapsed")
    ax.set_ylabel("Cumulative % of cost")
    ax.set_title("Normalized time:\nphasing shape is the ONLY driver",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    ax = axs[1]
    m = min(cfg.calendar_month, cfg.max_month)
    for dat, c, lab, al in [(sim.pct_fixed, SLATE, "shape fixed", 0.20),
                            (sim.pct_samp, TEAL, "shape sampled", 0.28)]:
        lo, hi = (np.percentile(dat, q, axis=0) for q in (5, 95))
        ax.fill_between(sim.months, lo * 100, hi * 100, color=c, alpha=al,
                        label=f"90% interval, {lab}")
    ax.plot(sim.months, np.percentile(sim.pct_samp, 50, axis=0) * 100,
            color=NAVY, lw=2.6, label="Median")
    w1 = (np.percentile(sim.pct_fixed[:, m], 95)
          - np.percentile(sim.pct_fixed[:, m], 5)) * 100
    w2 = (np.percentile(sim.pct_samp[:, m], 95)
          - np.percentile(sim.pct_samp[:, m], 5)) * 100
    ax.axvline(m, color=GOLD, ls="--", lw=1.3)
    ax.annotate(f"month {m}:\n{w1:.0f} pt \u2192 {w2:.0f} pt\n(duration dominates)",
                xy=(m, 50), xytext=(m + 12, 26), fontsize=10.5, color=GOLD,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative % of cost")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.set_title("Calendar time:\nduration uncertainty takes over",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_xlim(0, cfg.max_month)
    ax.set_ylim(0, 102)
    fig.tight_layout()
    _save(fig, out, "f04_coupling", inp)
    print(f" shape share of variance at month {m}: "
          f"{(1 - (w1 / w2) ** 2) * 100:.0f}%")


def fig05_mixture(inp: Inputs, cfg: SimConfig, out: Path, sim: SimResult) -> None:
    """A late fiscal year as a point mass at zero plus a continuous lobe."""
    y = min(cfg.mixture_fy, cfg.n_fy) - 1
    v = sim.fy[:, y]
    pz = float((v <= cfg.active_threshold).mean())
    lab = sim.fy_labels[y]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4),
                            gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axs[0]
    ax.bar(["$0\n(finished)", "> $0\n(still active)"], [pz, 1 - pz],
           color=[CRIM, TEAL], width=0.6)
    for i, val in enumerate([pz, 1 - pz]):
        ax.text(i, val + 0.02, f"{val:.0%}", ha="center", fontweight="bold",
                fontsize=13, color=[CRIM, TEAL][i])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of simulation runs")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Mixture weights", fontsize=11.5, fontweight="bold")
    ax.grid(False)

    ax = axs[1]
    pos = v[v > cfg.active_threshold]
    if pos.size:
        ax.hist(pos, bins=42, color=TEAL, alpha=0.9, edgecolor="white", lw=0.6)
        med = np.median(pos)
        ax.axvline(med, color=NAVY, lw=2.4, ls="--")
        ax.text(med, ax.get_ylim()[1] * 0.92, f" median ${med:.0f}M",
                color=NAVY, fontweight="bold", fontsize=11)
    ax.set_xlabel(f"{lab} spend ($M), runs still active")
    ax.set_ylabel("Iterations")
    ax.set_ylim(bottom=0)
    ax.set_title("Continuous part (conditional on active)",
                 fontsize=11.5, fontweight="bold")
    ax.grid(False)
    fig.suptitle(f"{lab} annual spend is a MIXTURE: point mass at $0 + continuous lobe",
                 fontsize=12.5, fontweight="bold", color=DEEP, y=0.995)
    lo, hi = cfg.duration_clip
    fig.text(0.5, 0.925,
             f"Notional duration: D ~ LogNormal(median={cfg.duration_median:.0f} mo, "
             f"σ={cfg.duration_sigma:.2f}), clipped to [{lo:.0f}, {hi:.0f}] mo",
             ha="center", fontsize=9.5, color=SLATE, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out, "f05_mixture", inp)


def fig06_quantile_bars(inp: Inputs, cfg: SimConfig, out: Path, sim: SimResult) -> None:
    """Unconditional annual quantiles, showing where the P50 profile truncates."""
    q15, q50, q85 = (np.percentile(sim.fy, q, axis=0) for q in (15, 50, 85))
    x = np.arange(cfg.n_fy)
    w = 1 / 3
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    ax.bar(x - w, q15, w, color=GOLD, label="P15")
    ax.bar(x, q50, w, color=NAVY, label="P50")
    ax.bar(x + w, q85, w, color=TEAL, label="P85")

    below = np.where(q50 < cfg.active_threshold)[0]
    if below.size:
        f0 = int(below[0])
        ax.axvspan(f0 - 0.5, cfg.n_fy - 0.5, color=CRIM, alpha=0.07)
        ax.annotate(f"P50 = $0 from {sim.fy_labels[f0]} onward,\n"
                    f"but P85 is still ${q85[f0]:.0f}M",
                    xy=(f0, q85[f0]), xytext=(max(f0 , 0.1), q85[f0] + 38),
                    fontsize=10.5, color=CRIM, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=CRIM, lw=1.6))
    ax.set_xticks(x)
    ax.set_xticklabels(sim.fy_labels)
    ax.set_ylabel("Annual spend ($M)")
    ax.set_title("Unconditional annual quantiles \u2014 the P50 profile truncates early",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out, "f06_quantile_bars", inp)

    tot_p80 = np.percentile(sim.cum[:, -1], 80)
    sum_q80 = np.percentile(sim.fy, 80, axis=0).sum()
    print(f" sum of annual P80s ${sum_q80:.0f}M vs total P80 ${tot_p80:.0f}M "
          f"(+{(sum_q80 / tot_p80 - 1) * 100:.0f}%)")


def fig07_conditional(inp: Inputs, cfg: SimConfig, out: Path, sim: SimResult) -> None:
    """Conditional annual spend given active, alongside P(active)."""
    n = cfg.n_fy
    x = np.arange(n)

    def cq(q):
        return np.array([np.percentile(sim.fy[sim.active[:, y], y], q)
                         if sim.active[:, y].sum() > 30 else np.nan
                         for y in range(n)])

    c15, c50, c85 = cq(15), cq(50), cq(85)
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x, c50, 0.75, color=TEAL, label="Median spend | active")
    ax.vlines(x, c15, c85, color=NAVY, lw=2.4, label="P15\u2013P85 | active")
    ax.set_ylabel("Annual spend given still active ($M)")
    ax.set_xticks(x)
    ax.set_xticklabels(sim.fy_labels)
    ax.grid(False)

    ax2 = ax.twinx()
    ax2.plot(x, sim.p_active, color=GOLD, lw=3, marker="o", ms=6, label="P(active)")
    ax2.set_ylabel("P(project still active)", color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.grid(False)
    for y in range(n):
        if 0.03 < sim.p_active[y] < 0.8:
            ax2.annotate(f"{sim.p_active[y]:.0%}", (y, sim.p_active[y]),
                         textcoords="offset points", xytext=(8, 10), ha="left",
                         color=GOLD, fontweight="bold", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("Report the two questions separately: will it run, "
                 "and how much if it does", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "f07_conditional", inp)


def fig08_fan_annotated(inp: Inputs, cfg: SimConfig, out: Path, sim: SimResult) -> None:
    """Cumulative fan chart with completion markers taken from the duration draws.

    Completion markers MUST come from the duration distribution. Do not infer them
    from where a cumulative-spend quantile flattens: the 95th percentile of
    cumulative dollars at month m is dominated by iterations spending fast and
    finishing EARLY, so the upper band flattens near the earliest completions,
    not the latest. Quantile of cumulative spend is not the cumulative spend of
    the quantile duration.
    """
    cum, months = sim.cum, sim.months
    q5, q25, q50, q75, q95 = (np.percentile(cum, q, axis=0)
                              for q in (5, 25, 50, 75, 95))
    d50 = float(np.median(sim.duration))
    d85 = float(np.percentile(sim.duration, 85))

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.fill_between(months, q5, q95, color=TEAL, alpha=0.18, label="90% interval")
    ax.fill_between(months, q25, q75, color=TEAL, alpha=0.34, label="50% interval")
    ax.plot(months, q50, color=NAVY, lw=2.8, label="Median cumulative spend")
    for k in range(min(9, cum.shape[0])):
        ax.plot(months, cum[k], color=SLATE, alpha=0.26, lw=0.9,
                label="Individual iterations" if k == 0 else None)
    ax.axvline(d50, color=NAVY, ls=":", lw=2.4,
               label=f"Median completion \u2014 month {d50:.0f}")
    ax.axvline(d85, color=GOLD, ls=":", lw=2.4,
               label=f"P85 completion \u2014 month {d85:.0f}")

    ytop = ax.get_ylim()[1]
    ya = ytop * 0.13
    ax.annotate("", xy=(d50, ya), xytext=(d85, ya),
                arrowprops=dict(arrowstyle="<->", color=CRIM, lw=2.2))
    ax.text((d50 + d85) / 2, ya * 1.28, f"schedule risk\n{d85 - d50:.0f} months",
            ha="center", color=CRIM, fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative spend ($M)")
    ax.set_xlim(0, cfg.max_month)
    ax.set_ylim(0, ytop)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "f08_fan_annotated", inp)


def fig09_per_formula(inp: Inputs, cfg: SimConfig, out: Path) -> None:
    """The phasing estimating relationship: formulas plus strip-area demonstration."""
    A, B = (float(np.exp(inp.mu_ln[0])), float(np.exp(inp.mu_ln[1])))
    ny = min(cfg.n_fy, 6)
    d_months = ny * cfg.fy_months
    edges = np.linspace(0, 1, ny + 1)
    fed = bcdf(edges, A, B)
    ann = np.diff(fed)

    fig = plt.figure(figsize=(13.4, 4.75))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.02, 1.0], height_ratios=[1.5, 1.0],
                  hspace=0.42, wspace=0.17, left=0.015, right=0.985,
                  top=0.90, bottom=0.115)

    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=MINT,
                               edgecolor=BORD, lw=1.2, zorder=0))
    T = ax.transAxes
    ax.text(.045, .94, "SPEND-TIMING DENSITY", fontsize=9.5, color=DTEAL,
            fontweight="bold", va="top", transform=T)
    ax.text(.045, .845,
            r"$f(t;\alpha,\beta)=\dfrac{t^{\alpha-1}(1-t)^{\beta-1}}{B(\alpha,\beta)}$"
            r"$,\qquad B(\alpha,\beta)=\dfrac{\Gamma(\alpha)\Gamma(\beta)}"
            r"{\Gamma(\alpha+\beta)}$",
            fontsize=15, color=DEEP, va="top", transform=T)
    ax.text(.045, .625, "CUMULATIVE PHASING", fontsize=9.5, color=DTEAL,
            fontweight="bold", va="top", transform=T)
    ax.text(.045, .535,
            r"$F(t;\alpha,\beta)=I_t(\alpha,\beta)=\dfrac{1}{B(\alpha,\beta)}"
            r"\int_0^{t}u^{\alpha-1}(1-u)^{\beta-1}\,du$",
            fontsize=15, color=DEEP, va="top", transform=T)
    ax.plot([.045, .955], [.40, .40], color=BORD, lw=1.4, transform=T)
    ax.text(.045, .345, "PHASING ESTIMATING RELATIONSHIP", fontsize=9.5, color=CRIM,
            fontweight="bold", va="top", transform=T)
    ax.text(.045, .245,
            r"$\mathrm{Spend}_y=C\left[F(t_y)-F(t_{y-1})\right]"
            r"=C\int_{t_{y-1}}^{t_y}f(u)\,du$",
            fontsize=16.5, color=CRIM, va="top", transform=T)
    ax.text(.045, .085,
            r"$t_y=m_y/D$" " \u2014 " r"$C$ = total cost draw, $D$ = duration draw,"
            "\n" r"$m_y$ = months elapsed at the end of year $y$",
            fontsize=11, color=SLATE, va="top", linespacing=1.7, transform=T)

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(1e-4, 1 - 1e-4, 500)
    pdf = sbeta.pdf(t, A, B)
    ax.plot(t, pdf, color=NAVY, lw=2.4, zorder=4)
    shades = np.interp(np.arange(ny), [0, (ny - 1) / 2, ny - 1], [.16, .44, .16])
    for k in range(ny):
        m = (t >= edges[k]) & (t <= edges[k + 1])
        ax.fill_between(t[m], 0, pdf[m], color=TEAL, alpha=float(shades[k]), zorder=2)
        ax.axvline(edges[k + 1], color=SLATE, lw=0.9, ls=":", alpha=0.8, zorder=3)
        xm = (edges[k] + edges[k + 1]) / 2
        ax.text(xm, sbeta.pdf(xm, A, B) * 0.42, f"{ann[k] * 100:.0f}%", ha="center",
                fontsize=11.5, fontweight="bold", color=DEEP, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, pdf.max() * 1.18)
    ax.set_xticks(edges)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("$f(t)$", fontsize=11)
    ax.set_title("Area of each strip = that year's phasing percentage",
                 fontsize=11.5, fontweight="bold", color=DEEP, pad=8)
    ax.grid(False)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(ny)
    ax.bar(xs, ann * 100, width=1.0, color=TEAL, edgecolor=NAVY, lw=1.4)
    for k, v in enumerate(ann * 100):
        ax.text(k, v + 1.1, f"{v:.0f}%", ha="center", fontsize=10.5,
                fontweight="bold", color=DEEP)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"FY{k + 1}" for k in xs], fontsize=10)
    ax.set_ylabel("% of total", fontsize=10.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.set_ylim(0, max(ann * 100) * 1.32)
    ax.text(.985, .86, f"sums to {ann.sum() * 100:.0f}%", transform=ax.transAxes,
            ha="right", fontsize=10.5, color=CRIM, fontweight="bold")
    ax.grid(False)

    fig.suptitle(f"Worked example: $\\alpha$={A:.2f}, $\\beta$={B:.2f}, "
                 f"D={d_months} months ({ny} fiscal years of equal length)",
                 fontsize=11, color=SLATE, y=1.03)
    _save(fig, out, "f09_per_formula", inp, tight=True)


# ============================================================================
# Driver
# ============================================================================

FIGURES = {
    "f01": "curve fan", "f02": "beta atlas", "f03": "scatter + ellipses",
    "f03a": "scatter + ellipses (mu/nu)",
    "f04": "coupling", "f05": "mixture", "f06": "quantile bars",
    "f07": "conditional", "f08": "fan annotated", "f09": "PER formula",
}
NEEDS_SIM = {"f04", "f05", "f06", "f07", "f08"}


def generate_figures(inp: Inputs, cfg: SimConfig, out: Path,
                     rng: np.random.Generator | None = None,
                     only: set[str] | None = None) -> None:
    """Render the requested deck figures (all of them by default) into `out`.

    Shared by the standalone CLI below and the pipeline's --presentation
    stage. The deck's global matplotlib style is applied inside an
    rc_context so it does not leak into figures the caller renders later.
    """
    out.mkdir(parents=True, exist_ok=True)
    rng = rng or np.random.default_rng(cfg.seed)

    print(f"stratum : {inp.cost_type}")
    print(f"curves : {len(inp.alpha)}")
    print(f"population mu : ln alpha {inp.mu_ln[0]:+.4f}, ln beta {inp.mu_ln[1]:+.4f}")
    sd = np.sqrt(np.diag(inp.cov_ln))
    rho = inp.cov_ln[0, 1] / (sd[0] * sd[1]) if sd.prod() > 0 else 0.0
    print(f"population sd : {sd[0]:.4f}, {sd[1]:.4f} rho {rho:+.4f}")
    for n in inp.notes:
        print(f" note: {n}")
    if inp.illustrative:
        print("\n *** ILLUSTRATIVE MODE -- figures will be watermarked ***")
        print(" *** supply --fits and --dists to use fitted values ***")
    print()

    wanted = set(only) if only else set(FIGURES)
    sim = None
    if wanted & NEEDS_SIM:
        print(f"simulating {cfg.n_iter:,} iterations ...")
        sim = simulate(inp, cfg, rng)
        print(f" median duration {np.median(sim.duration):.0f} mo, "
              f"P95 {np.percentile(sim.duration, 95):.0f} mo")
        print(" P(active): " + " ".join(
            f"{l}={v:.2f}" for l, v in zip(sim.fy_labels, sim.p_active)))
        print()

    with plt.rc_context():
        apply_style()
        print("figures:")
        if "f01" in wanted:
            fig01_curve_fan(inp, cfg, out)
        if "f02" in wanted:
            fig02_beta_atlas(inp, cfg, out)
        if "f03" in wanted:
            fig03_scatter_ellipse(inp, cfg, out)
        if "f03a" in wanted:
            fig03a_scatter_ellipse_munu(inp, cfg, out)
        if "f04" in wanted:
            fig04_coupling(inp, cfg, out, sim)
        if "f05" in wanted:
            fig05_mixture(inp, cfg, out, sim)
        if "f06" in wanted:
            fig06_quantile_bars(inp, cfg, out, sim)
        if "f07" in wanted:
            fig07_conditional(inp, cfg, out, sim)
        if "f08" in wanted:
            fig08_fan_annotated(inp, cfg, out, sim)
        if "f09" in wanted:
            fig09_per_formula(inp, cfg, out)

    print(f"\ndone -> {out.resolve()}")
    if inp.illustrative:
        print("REMINDER: these figures are watermarked ILLUSTRATIVE.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits", help="fits.csv from phasing_analysis.py")
    p.add_argument("--dists", help="distributions.json from phasing_analysis.py")
    p.add_argument("--cost-type", default="TPC", help="stratum to feature (default TPC)")
    p.add_argument("--out", default="presentation_figures",
                   help="output directory (default presentation_figures/)")
    p.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                   help="generate only these figures")
    p.add_argument("--duration-median", type=float, default=SimConfig.duration_median)
    p.add_argument("--duration-sigma", type=float, default=SimConfig.duration_sigma)
    p.add_argument("--cost-median", type=float, default=SimConfig.cost_median)
    p.add_argument("--cost-sigma", type=float, default=SimConfig.cost_sigma)
    p.add_argument("--n-iter", type=int, default=SimConfig.n_iter)
    p.add_argument("--n-fy", type=int, default=SimConfig.n_fy)
    p.add_argument("--max-month", type=int, default=SimConfig.max_month)
    p.add_argument("--mixture-fy", type=int, default=SimConfig.mixture_fy)
    p.add_argument("--seed", type=int, default=SimConfig.seed)
    args = p.parse_args()

    cfg = SimConfig(duration_median=args.duration_median,
                    duration_sigma=args.duration_sigma,
                    cost_median=args.cost_median, cost_sigma=args.cost_sigma,
                    n_iter=args.n_iter, n_fy=args.n_fy, max_month=args.max_month,
                    mixture_fy=args.mixture_fy, seed=args.seed)
    rng = np.random.default_rng(cfg.seed)

    print("=" * 70)
    print("PHASING DECK FIGURES")
    print("=" * 70)
    inp = load_inputs(args.fits, args.dists, args.cost_type, rng)
    generate_figures(inp, cfg, Path(args.out), rng=rng,
                     only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()