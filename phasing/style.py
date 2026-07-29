"""Small plotting-style helpers shared by `eda.py` and `correlation.py`.

Split out so those two peer modules don't have to import from each other.
"""

from __future__ import annotations

import pandas as pd

from . import config


def color_for(ct: str, i: int):
    return config.COLORS.get(ct, config.FALLBACK_COLORS[i % 10])


def ordered_types(fits: pd.DataFrame) -> list[str]:
    present = list(fits["cost_type"].unique())
    ordered = [c for c in config.COST_TYPE_ORDER if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered
