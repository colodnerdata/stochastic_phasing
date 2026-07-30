"""STAGE 1 — LOAD & VALIDATE.

Loads `data.csv`, resolves flexible column names, coerces/rescales numerics,
and sorts curves into chronological order. Monotonicity screening lives in
`phasing.screening`, not here — this module is purely mechanical.

Two input formats are auto-detected, based on which columns are present:
  cumulative:  project, cost_type, year, cum_pct_cost, cum_pct_schedule
  incremental: project, year_from_start, inc_pct_spend, cum_pct_sched
    (no cost_type column; treated as a single undifferentiated spend curve
    per project, labeled config.DEFAULT_COST_TYPE; cum_pct_cost is derived
    from inc_pct_spend via a within-curve cumulative sum)
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from . import config


def _normalize_colname(c: str) -> str:
    """'Inc. Pct. Spend' -> 'inc_pct_spend'; collapse any run of
    non-alphanumeric characters (spaces, periods, %, #, ...) to a single
    underscore so headers like 'Cum. Pct Sched' match 'cum_pct_sched'."""
    s = re.sub(r"[^0-9a-z]+", "_", c.strip().lower())
    return s.strip("_")


def _resolve_columns(df: pd.DataFrame) -> dict:
    """Map logical names -> actual column names, honoring overrides.

    Two input formats are supported, auto-detected by which columns are
    present (no CLI flag needed):
      - cumulative: project, cost_type, pct_sched, cum_pct_cost [, year]
      - incremental: project, [year_from_start,] inc_pct_cost, cum_pct_sched
        (no cost_type column; every row is a single undifferentiated spend
        curve per project -- see config.DEFAULT_COST_TYPE)
    pct_cost (cumulative) is preferred whenever both are resolvable.
    """
    norm = {c: _normalize_colname(c) for c in df.columns}

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
            contains_none=("sched", "type", "inc"),
        )
        or find(set(), contains_all=("cost",), contains_none=("sched", "type", "inc")),
        "year": config.COL_OVERRIDES["year"]
        or find({
            "year", "fy", "fiscal_year", "yr",
            "year_from_start", "years_from_start", "yr_from_start",
            "years_since_start", "elapsed_years", "project_year",
        })
        or find(set(), contains_all=("year", "start")),
    }
    resolved["inc_pct_cost"] = None
    if resolved["pct_cost"] is None:
        # Cumulative cost column not found -- fall back to an incremental
        # (per-period) spend column, converted to cumulative in
        # load_and_validate via a within-curve cumsum.
        resolved["inc_pct_cost"] = config.COL_OVERRIDES["inc_pct_cost"] or find(
            {"inc_pct_cost", "inc_pct_spend", "incremental_pct_cost",
             "incremental_pct_spend", "inc_cost_pct", "inc_spend_pct"},
            contains_all=("inc", "cost"), contains_none=("cum",),
        ) or find(
            set(), contains_all=("inc", "spend"), contains_none=("cum",),
        )

    # 'year' and 'cost_type' are optional -- everything else is required,
    # and exactly one of pct_cost / inc_pct_cost must resolve.
    required = {k: v for k, v in resolved.items()
                if k not in ("year", "cost_type", "pct_cost", "inc_pct_cost")}
    missing = [k for k, v in required.items() if v is None]
    if resolved["pct_cost"] is None and resolved["inc_pct_cost"] is None:
        missing.append("pct_cost (cumulative) or inc_pct_cost (incremental)")
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
              "Place data.csv next to the repo root, or pass --data <path>.")
        sys.exit(1)

    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    cols = _resolve_columns(df)
    print(f"Column mapping: {cols}")

    rename_map = {
        cols["project"]: "project",
        cols["pct_sched"]: "pct_sched",
    }
    if cols["cost_type"]:
        rename_map[cols["cost_type"]] = "cost_type"
    if cols["pct_cost"]:
        rename_map[cols["pct_cost"]] = "pct_cost"
    else:
        rename_map[cols["inc_pct_cost"]] = "inc_pct_cost"
    if cols["year"]:
        rename_map[cols["year"]] = "year"
        print(f"  Year column detected: '{cols['year']}' "
              "(used for chronological sort + vintage drift check)")
    else:
        print("  No year column detected — sorting by pct_sched only")
    df = df.rename(columns=rename_map)

    if "cost_type" not in df.columns:
        df["cost_type"] = config.DEFAULT_COST_TYPE
        print(f"  No cost_type column detected — treating all rows as a "
              f"single cost type ('{config.DEFAULT_COST_TYPE}')")
    df["cost_type"] = df["cost_type"].astype(str).str.strip().str.upper()

    if "pct_cost" in df.columns:
        # Coerce numerics; rescale 0-100 -> 0-1 if needed
        for c in ("pct_sched", "pct_cost"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].max() > 1.5:
                print(f"  {c}: values exceed 1.5 -> assuming percent, dividing by 100")
                df[c] = df[c] / 100.0
        spend_cols = ["pct_sched", "pct_cost"]
    else:
        print(f"  Incremental spend column detected: '{cols['inc_pct_cost']}' "
              "-> will derive cumulative pct_cost via within-curve cumsum")
        df["pct_sched"] = pd.to_numeric(df["pct_sched"], errors="coerce")
        if df["pct_sched"].max() > 1.5:
            print("  pct_sched: values exceed 1.5 -> assuming percent, dividing by 100")
            df["pct_sched"] = df["pct_sched"] / 100.0
        df["inc_pct_cost"] = pd.to_numeric(df["inc_pct_cost"], errors="coerce")
        spend_cols = ["pct_sched", "inc_pct_cost"]

    n_bad = df[spend_cols].isna().any(axis=1).sum()
    if n_bad:
        print(f"  Dropping {n_bad} rows with non-numeric pct values")
        df = df.dropna(subset=spend_cols)

    sort_cols = ["project", "cost_type"]
    if "year" in df.columns:
        sort_cols.append("year")   # true chronology first; robust to tied pct_sched
    sort_cols.append("pct_sched")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if "pct_cost" not in df.columns:
        # Incremental format: cumulate within each curve, in the same
        # chronological order just established by the sort above.
        df["pct_cost"] = df.groupby(["project", "cost_type"])["inc_pct_cost"].cumsum()
        if df["pct_cost"].max() > 1.5:
            print("  pct_cost (cumsum of incremental spend): values exceed "
                  "1.5 -> assuming percent, dividing by 100")
            df["pct_cost"] = df["pct_cost"] / 100.0

    print("\nDataset summary:")
    print(f"  Projects:   {df['project'].nunique()}")
    print(f"  Cost types: {sorted(df['cost_type'].unique())}")
    pts = df.groupby(["project", "cost_type"]).size()
    print(f"  Points per curve: min={pts.min()}, median={pts.median():.0f}, "
          f"max={pts.max()}")

    return df, cols
