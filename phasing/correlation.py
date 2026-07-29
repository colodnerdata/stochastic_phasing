"""STAGE 4 — CORRELATION & NORMALITY.

Operates on `fits_for_stage2` (bound-contact fits already excluded by
`phasing.screening.screen_post_fit`) since these statistics justify the
Stage 5 joint-distribution model, and a degenerate boundary fit would
bias exactly the assumption this stage exists to validate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, pearsonr, shapiro, spearmanr

from . import config
from .style import color_for as _color_for, ordered_types as _ordered_types


def correlation_analysis(fits_for_stage2: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STAGE 4: CORRELATION & NORMALITY")
    print("=" * 70)

    rows = []
    for ct, sub in fits_for_stage2.groupby("cost_type"):
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
    corr.to_csv(config.OUTPUT_DIR / config.CORRELATIONS_FILENAME, index=False)
    return corr


def plot_correlation_panels(fits_for_stage2: pd.DataFrame):
    types = _ordered_types(fits_for_stage2)
    fig, axes = plt.subplots(len(types), 2, figsize=(12, 4.5 * len(types)),
                             squeeze=False)
    for i, ct in enumerate(types):
        sub = fits_for_stage2[fits_for_stage2["cost_type"] == ct]
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
    fig.savefig(config.OUTPUT_DIR / "03_correlation_panels.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_qq_normality(fits_for_stage2: pd.DataFrame):
    """Mahalanobis distance^2 vs chi2(2) quantiles, unit vs log space."""
    types = _ordered_types(fits_for_stage2)
    fig, axes = plt.subplots(len(types), 2, figsize=(12, 4.2 * len(types)),
                             squeeze=False)
    for i, ct in enumerate(types):
        sub = fits_for_stage2[fits_for_stage2["cost_type"] == ct]
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
    fig.savefig(config.OUTPUT_DIR / "04_qq_normality.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
