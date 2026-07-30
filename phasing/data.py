"""STAGE 1 — LOAD & VALIDATE.

Loads `data.csv`, resolves flexible column names, coerces/rescales numerics,
and sorts curves into chronological order. Monotonicity screening lives in
`phasing.screening`, not here — this module is purely mechanical.
"""

from __future__ import annotations

import sys

import pandas as pd

from . import config


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
        "project": config.COL_OVERRIDES["project"]
        or find({"project", "project_id", "proj", "program"}),
        "cost_type": config.COL_OVERRIDES["cost_type"]
        or find({"cost_type", "costtype", "type", "cost_category"}),
        "pct_sched": config.COL_OVERRIDES["pct_sched"]
        or find(
            {"cum_pct_schedule", "cum_pct_sched", "pct_schedule", "pct_sched"},
            contains_all=("sched",),
        )
        or find(set(), contains_all=("time",), contains_none=("cost",)),
        "pct_cost": config.COL_OVERRIDES["pct_cost"]
        or find(
            {"cum_pct_cost", "pct_cost", "cum_cost_pct"},
            contains_all=("cost", "pct"),
            contains_none=("sched", "type"),
        )
        or find(set(), contains_all=("cost",), contains_none=("sched", "type")),
        "year": config.COL_OVERRIDES["year"]
        or find({"year", "fy", "fiscal_year", "yr"}),
    }

    # 'year' is optional — everything else is required
    missing = [k for k, v in resolved.items() if v is None and k != "year"]
    if missing:
        print(f"\nERROR: could not resolve columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        print("Set COL_OVERRIDES at the top of phasing/config.py and re-run.")
        sys.exit(1)
    return resolved


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("STAGE 1: LOAD & VALIDATE")
    print("=" * 70)

    if not config.DATA_PATH.exists():
        print(f"ERROR: {config.DATA_PATH} not found. "
              "Place data.csv next to the repo root.")
        sys.exit(1)

    df = pd.read_csv(config.DATA_PATH)
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

    print("\nDataset summary:")
    print(f"  Projects:   {df['project'].nunique()}")
    print(f"  Cost types: {sorted(df['cost_type'].unique())}")
    pts = df.groupby(["project", "cost_type"]).size()
    print(f"  Points per curve: min={pts.min()}, median={pts.median():.0f}, "
          f"max={pts.max()}")

    return df, cols
