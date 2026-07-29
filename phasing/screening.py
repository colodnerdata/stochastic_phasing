"""Data-quality screening.

Owns every exclusion decision in the pipeline, in one auditable place:

  * PRE-FIT:  monotonicity violations (known before any curve is fit) —
              the whole (project, cost_type) curve is dropped before
              `fitting.fit_all_curves` ever sees it.
  * AT-FIT:   insufficient interior points — decided inside
              `fitting.fit_all_curves` itself (it owns the point-prep
              logic), surfaced here via the `status` column it sets.
  * POST-FIT: bound-contact fits (only knowable once the optimizer has
              run) — excluded from Stage 2 (joint-distribution fit +
              PhasingPredictor) but kept in fits.csv for review.

`build_screening_report` stitches all three reasons into one tidy
DataFrame covering every (project, cost_type) group ever loaded, for a
single audit trail regardless of which stage excluded it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def detect_monotonicity_violations(
    df: pd.DataFrame, tol: float = config.MONOTONICITY_TOL
) -> pd.DataFrame:
    """Per (project, cost_type) group, find violations of monotone
    non-decreasing pct_sched / pct_cost beyond float tolerance `tol`.

    Returns one row per violating group: project, cost_type,
    n_points_total, reason, detail.
    """
    records = []
    for (proj, ct), g in df.groupby(["project", "cost_type"]):
        details = []
        for col in ("pct_sched", "pct_cost"):
            diffs = np.diff(g[col].values)
            bad = np.where(diffs < -tol)[0]
            if len(bad):
                worst = bad[np.argmin(diffs[bad])]
                details.append(
                    f"{col} decreased by {abs(diffs[worst]):.4g} "
                    f"at interior index {worst}"
                )
        if details:
            records.append(dict(
                project=proj, cost_type=ct,
                n_points_total=len(g),
                reason="monotonicity_violation",
                detail="; ".join(details),
            ))
    return pd.DataFrame(
        records,
        columns=["project", "cost_type", "n_points_total", "reason", "detail"],
    )


def screen_pre_fit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop (project, cost_type) groups with a monotonicity violation.

    Returns (df_clean, pre_fit_exclusions).
    """
    print("\nMonotonicity screen:")
    violations = detect_monotonicity_violations(df)
    n_curves = df.groupby(["project", "cost_type"]).ngroups
    if violations.empty:
        print(f"  {n_curves} curves checked, 0 excluded")
        return df, violations

    bad_keys = set(zip(violations["project"], violations["cost_type"]))
    for proj, ct in sorted(bad_keys):
        detail = violations.loc[
            (violations["project"] == proj) & (violations["cost_type"] == ct),
            "detail",
        ].iloc[0]
        print(f"  EXCLUDE {proj}/{ct}: {detail}")

    keep_mask = ~df.set_index(["project", "cost_type"]).index.isin(bad_keys)
    df_clean = df[keep_mask].reset_index(drop=True)
    print(f"  {n_curves} curves checked, {len(bad_keys)} excluded "
          f"for monotonicity violations")
    return df_clean, violations


def screen_post_fit(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split fitted curves into the Stage-2-eligible set and the
    bound-contact exclusions.

    Returns (fits_for_stage2, post_fit_exclusions).
    """
    fitted = fits[fits["status"] == "fit"]
    at_bound = fitted[fitted["at_bound"]]
    fits_for_stage2 = fitted[~fitted["at_bound"]].reset_index(drop=True)

    exclusions = pd.DataFrame({
        "project": at_bound["project"],
        "cost_type": at_bound["cost_type"],
        "n_points_total": at_bound["n_points"],
        "reason": "bound_contact",
        "detail": at_bound.apply(
            lambda r: f"alpha={r['alpha']:.4g}, beta={r['beta']:.4g} "
                      f"at fit bound {config.FIT_BOUNDS}",
            axis=1,
        ) if len(at_bound) else pd.Series(dtype=str),
    }).reset_index(drop=True)

    print(f"\nBound-contact screen: {len(fitted)} fits checked, "
          f"{len(at_bound)} excluded from Stage 2 (kept in fits.csv)")
    return fits_for_stage2, exclusions


def build_screening_report(
    df_raw: pd.DataFrame,
    pre_fit_exclusions: pd.DataFrame,
    fits: pd.DataFrame,
    post_fit_exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (project, cost_type) group ever seen in the raw data."""
    raw_groups = df_raw.groupby(["project", "cost_type"]).agg(
        n_points_total=("pct_sched", "size"),
        year=("year", "min") if "year" in df_raw.columns else ("pct_sched", "size"),
    ).reset_index()
    if "year" not in df_raw.columns:
        raw_groups["year"] = np.nan

    rows = []
    pre_fit_keys = set(zip(pre_fit_exclusions["project"], pre_fit_exclusions["cost_type"]))
    post_fit_keys = set(zip(post_fit_exclusions["project"], post_fit_exclusions["cost_type"]))
    fits_idx = fits.set_index(["project", "cost_type"])

    for _, g in raw_groups.iterrows():
        key = (g["project"], g["cost_type"])
        base = dict(
            project=g["project"], cost_type=g["cost_type"], year=g["year"],
            n_points_total=int(g["n_points_total"]),
        )

        if key in pre_fit_keys:
            detail = pre_fit_exclusions.loc[
                (pre_fit_exclusions["project"] == key[0])
                & (pre_fit_exclusions["cost_type"] == key[1]),
                "detail",
            ].iloc[0]
            rows.append(dict(
                **base, n_interior_points=np.nan,
                status="excluded", exclusion_stage="pre_fit",
                reason="monotonicity_violation", detail=detail,
                included_in_stage2=False, fit_r2=np.nan, fit_rmse=np.nan,
            ))
            continue

        if key not in fits_idx.index:
            # Shouldn't happen (every non-excluded group is attempted),
            # but keep the report complete if it ever does.
            rows.append(dict(
                **base, n_interior_points=np.nan,
                status="excluded", exclusion_stage="fit",
                reason="not_attempted", detail="",
                included_in_stage2=False, fit_r2=np.nan, fit_rmse=np.nan,
            ))
            continue

        fit_row = fits_idx.loc[key]
        if isinstance(fit_row, pd.DataFrame):  # defensive: duplicate keys
            fit_row = fit_row.iloc[0]

        if fit_row["status"] == "insufficient_points":
            rows.append(dict(
                **base, n_interior_points=int(fit_row["n_fit"]),
                status="excluded", exclusion_stage="fit",
                reason="insufficient_interior_points",
                detail=f"n_interior={int(fit_row['n_fit'])} "
                       f"< MIN_INTERIOR_POINTS={config.MIN_INTERIOR_POINTS}",
                included_in_stage2=False, fit_r2=np.nan, fit_rmse=np.nan,
            ))
        elif key in post_fit_keys:
            detail = post_fit_exclusions.loc[
                (post_fit_exclusions["project"] == key[0])
                & (post_fit_exclusions["cost_type"] == key[1]),
                "detail",
            ].iloc[0]
            rows.append(dict(
                **base, n_interior_points=int(fit_row["n_fit"]),
                status="excluded", exclusion_stage="post_fit",
                reason="bound_contact", detail=detail,
                included_in_stage2=False,
                fit_r2=float(fit_row["r2"]), fit_rmse=float(fit_row["rmse"]),
            ))
        else:
            rows.append(dict(
                **base, n_interior_points=int(fit_row["n_fit"]),
                status="clean", exclusion_stage="none",
                reason="none", detail="",
                included_in_stage2=True,
                fit_r2=float(fit_row["r2"]), fit_rmse=float(fit_row["rmse"]),
            ))

    return pd.DataFrame(rows, columns=[
        "project", "cost_type", "year", "n_points_total", "n_interior_points",
        "status", "exclusion_stage", "reason", "detail", "included_in_stage2",
        "fit_r2", "fit_rmse",
    ])


def write_screening_report(report_df: pd.DataFrame, path) -> None:
    report_df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def screening_summary_counts(report_df: pd.DataFrame) -> dict:
    counts = report_df["reason"].value_counts().to_dict()
    excluded = report_df[report_df["status"] == "excluded"]
    return {
        "n_total_groups": len(report_df),
        "n_clean": int((report_df["status"] == "clean").sum()),
        "n_monotonicity_violation": int(counts.get("monotonicity_violation", 0)),
        "n_insufficient_interior_points": int(counts.get("insufficient_interior_points", 0)),
        "n_bound_contact": int(counts.get("bound_contact", 0)),
        "excluded_pairs": list(zip(excluded["project"], excluded["cost_type"],
                                    excluded["reason"])),
    }
