"""Small plotting-style helpers shared by `eda.py`, `correlation.py`, and
`presentation_figures.py`.

Split out so those peer modules don't have to import from each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from . import config


def color_for(ct: str, i: int):
    return config.COLORS.get(ct, config.FALLBACK_COLORS[i % 10])


def conf_ellipse(ax, x, y, color, q: float = 0.95, **kw) -> None:
    """Add a q-level confidence ellipse under a bivariate normal approximation."""
    if len(x) < 3:
        return
    d = np.column_stack([x, y])
    mean, cov = d.mean(axis=0), np.cov(d.T)
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    k = chi2.ppf(q, df=2)
    ax.add_patch(Ellipse(mean, 2 * np.sqrt(k * evals[0]), 2 * np.sqrt(k * evals[1]),
                         angle=angle, facecolor=color, alpha=0.12, edgecolor=color,
                         lw=2, ls="--", **kw))


def ordered_types(fits: pd.DataFrame) -> list[str]:
    present = list(fits["cost_type"].unique())
    ordered = [c for c in config.COST_TYPE_ORDER if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered
