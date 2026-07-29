"""STAGE 3 — EDA.

Plotting functions here receive the *full* set of successfully-fitted
curves (bound-contact fits included) rather than the Stage-2-screened
subset, so a reviewer can see what a bound-contact curve actually looks
like — not just that it was excluded. Bound-contact curves are visually
distinguished (dashed line / hollow marker) wherever they appear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from . import config
from .fitting import beta_cdf
from .style import color_for as _color_for, ordered_types as _ordered_types


def plot_curve_overlays(df: pd.DataFrame, fits: pd.DataFrame):
    fitted = fits[fits["status"] == "fit"]
    types = _ordered_types(fitted)
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 5.5),
                             squeeze=False)
    t_plot = np.linspace(0, 1, 300)

    for i, ct in enumerate(types):
        ax = axes[0][i]
        sub = fitted[fitted["cost_type"] == ct]
        col = _color_for(ct, i)
        # raw data as faint dots
        raw = df[df["cost_type"] == ct]
        ax.plot(raw["pct_sched"], raw["pct_cost"], ".", ms=2, alpha=0.15,
                color="gray")

        clean = sub[~sub["at_bound"]]
        bound = sub[sub["at_bound"]]
        for _, r in clean.iterrows():
            ax.plot(t_plot, beta_cdf(t_plot, r["alpha"], r["beta"]),
                    color=col, alpha=0.35, lw=1.2)
        for j, (_, r) in enumerate(bound.iterrows()):
            ax.plot(t_plot, beta_cdf(t_plot, r["alpha"], r["beta"]),
                    color=col, alpha=0.6, lw=1.5, linestyle=":",
                    label="Bound-contact (excluded from Stage 2)" if j == 0 else None)
        # curve at the mean parameters (of the Stage-2-eligible population)
        if len(clean):
            ax.plot(t_plot, beta_cdf(t_plot, clean["alpha"].mean(), clean["beta"].mean()),
                    color="black", lw=2.5, label="Curve at mean (ᾱ, β̄)")
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Linear phasing")
        ax.set_title(f"{ct}  (n={len(sub)} projects, "
                     f"{len(bound)} bound-contact)", fontweight="bold")
        ax.set_xlabel("Cumulative % Schedule")
        ax.set_ylabel("Cumulative % Cost")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("Fitted Beta CDF Phasing Curves by Cost Type",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.OUTPUT_DIR / "01_curve_overlays.png", dpi=150,
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
    fitted = fits[fits["status"] == "fit"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    types = _ordered_types(fitted)

    for space, ax in zip(("unit", "log"), axes):
        for i, ct in enumerate(types):
            sub = fitted[fitted["cost_type"] == ct]
            clean = sub[~sub["at_bound"]]
            bound = sub[sub["at_bound"]]
            col = _color_for(ct, i)

            x = clean["alpha"].values
            y = clean["beta"].values
            if space == "log":
                x, y = np.log(x), np.log(y)
            ax.scatter(x, y, s=90, color=col, edgecolor="black", lw=1,
                       alpha=0.8, label=f"{ct} (n={len(clean)})")
            _add_conf_ellipse(ax, x, y, col)

            if len(bound):
                bx = bound["alpha"].values
                by = bound["beta"].values
                if space == "log":
                    bx, by = np.log(bx), np.log(by)
                ax.scatter(bx, by, s=110, facecolor="none", edgecolor=col,
                           lw=2, linestyle="--", marker="D",
                           label=f"{ct} bound-contact (n={len(bound)}, excluded)")
        if space == "unit":
            ax.set_xlabel("α"); ax.set_ylabel("β")
            ax.set_title("Unit space", fontweight="bold")
        else:
            ax.set_xlabel("ln α"); ax.set_ylabel("ln β")
            ax.set_title("Log space", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Fitted (α, β) with 95% Confidence Ellipses "
                 "(ellipses fit to Stage-2-eligible points only)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.OUTPUT_DIR / "02_scatter_ellipses.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
