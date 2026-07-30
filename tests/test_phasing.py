"""
Tests for the mean-precision (mu, nu) parameterization and the
incremental-spend input format added to the `phasing` package. Covers the
acceptance criteria from the feature spec: round-trip bijection,
delta-method vs numerical Jacobian, parameterization invariance of the
fitted curves, bvn_munu sampling, and the new input-format column
resolution / cumulative-cost derivation.

Run with: pytest tests/test_phasing.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from phasing import (
    config,
    correlation,
    data,
    distributions,
    fitting,
    screening,
    verification,
)
from phasing.fitting import beta_cdf
from phasing.parameterization import (
    ab_to_mu_nu,
    delta_method_cov,
    jacobian_ab_to_mu_nu,
    mu_nu_to_ab,
)

# ============================================================================
# Helpers
# ============================================================================

def _make_synthetic_df(seed=7, n_curves=6, n_pts=10, cost_type="TPC"):
    """Small synthetic (project, cost_type, pct_sched, pct_cost) dataset
    with known (mu, nu), for fitting/verification tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_curves):
        mu_true = rng.uniform(0.3, 0.7)
        nu_true = rng.uniform(4, 20)
        a, b = mu_nu_to_ab(mu_true, nu_true)
        t = np.sort(rng.uniform(0.02, 0.98, n_pts))
        y = beta_cdf(t, a, b)
        for tt, yy in zip(t, y):
            rows.append(dict(project=f"P{i}", cost_type=cost_type,
                              pct_sched=float(tt), pct_cost=float(yy)))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _isolated_output_dir(tmp_path, monkeypatch):
    """Redirect OUTPUT_DIR so tests never touch the real outputs/ folder."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)


# ============================================================================
# 3. Round-trip: mu_nu_to_ab(*ab_to_mu_nu(a, b)) recovers (a, b)
# ============================================================================

def test_round_trip_ab_to_mu_nu_grid():
    a_vals = np.linspace(0.1, 50.0, 30)
    b_vals = np.linspace(0.1, 50.0, 30)
    alpha, beta = np.meshgrid(a_vals, b_vals)

    mu, nu = ab_to_mu_nu(alpha, beta)
    alpha2, beta2 = mu_nu_to_ab(mu, nu)

    assert np.allclose(alpha, alpha2, rtol=1e-12)
    assert np.allclose(beta, beta2, rtol=1e-12)
    assert np.all((mu > 0) & (mu < 1))
    assert np.all(nu > 0)


def test_round_trip_scalar():
    for a, b in [(0.1, 0.1), (50.0, 50.0), (0.1, 50.0), (12.3, 4.5)]:
        mu, nu = ab_to_mu_nu(a, b)
        a2, b2 = mu_nu_to_ab(mu, nu)
        assert a2 == pytest.approx(a, rel=1e-12)
        assert b2 == pytest.approx(b, rel=1e-12)


def test_ab_to_mu_nu_rejects_invalid_domain():
    with pytest.raises(ValueError):
        ab_to_mu_nu(0.0, 1.0)
    with pytest.raises(ValueError):
        ab_to_mu_nu(-1.0, 1.0)
    with pytest.raises(ValueError):
        ab_to_mu_nu(np.nan, 1.0)


def test_mu_nu_to_ab_rejects_invalid_domain():
    with pytest.raises(ValueError):
        mu_nu_to_ab(0.0, 5.0)      # mu must be strictly > 0
    with pytest.raises(ValueError):
        mu_nu_to_ab(1.0, 5.0)      # mu must be strictly < 1
    with pytest.raises(ValueError):
        mu_nu_to_ab(0.5, -1.0)     # nu must be > 0
    with pytest.raises(ValueError):
        mu_nu_to_ab(0.5, np.inf)   # must be finite


# ============================================================================
# 4. Delta method vs numerical Jacobian (scipy.optimize.approx_fprime)
# ============================================================================

def test_jacobian_matches_numerical_derivative():
    rng = np.random.default_rng(0)
    eps = np.sqrt(np.finfo(float).eps)
    for _ in range(25):
        a, b = rng.uniform(0.2, 40.0, size=2)

        def mu_of(x):
            mu, _ = ab_to_mu_nu(x[0], x[1])
            return mu

        def nu_of(x):
            _, nu = ab_to_mu_nu(x[0], x[1])
            return nu

        j_num = np.vstack([
            approx_fprime(np.array([a, b]), mu_of, eps),
            approx_fprime(np.array([a, b]), nu_of, eps),
        ])
        j_ana = jacobian_ab_to_mu_nu(a, b)
        assert np.allclose(j_ana, j_num, rtol=1e-5, atol=1e-8)


def test_delta_method_cov_matches_numerical_jacobian():
    rng = np.random.default_rng(1)
    eps = np.sqrt(np.finfo(float).eps)
    cov_ab = np.array([[1.3, 0.4], [0.4, 0.9]])  # arbitrary SPD covariance
    for _ in range(10):
        a, b = rng.uniform(0.2, 40.0, size=2)

        def mu_of(x):
            mu, _ = ab_to_mu_nu(x[0], x[1])
            return mu

        def nu_of(x):
            _, nu = ab_to_mu_nu(x[0], x[1])
            return nu

        j_num = np.vstack([
            approx_fprime(np.array([a, b]), mu_of, eps),
            approx_fprime(np.array([a, b]), nu_of, eps),
        ])
        cov_num = j_num @ cov_ab @ j_num.T
        cov_ana = delta_method_cov(cov_ab, a, b)
        assert np.allclose(cov_ana, cov_num, rtol=1e-5)


def test_jacobian_batched_shape():
    alpha = np.array([1.0, 2.0, 3.0])
    beta = np.array([4.0, 5.0, 6.0])
    J = jacobian_ab_to_mu_nu(alpha, beta)
    assert J.shape == (3, 2, 2)
    for i in range(3):
        assert np.allclose(J[i], jacobian_ab_to_mu_nu(alpha[i], beta[i]))


def test_jacobian_rejects_invalid_domain():
    with pytest.raises(ValueError):
        jacobian_ab_to_mu_nu(0.0, 1.0)
    with pytest.raises(ValueError):
        jacobian_ab_to_mu_nu(-1.0, 1.0)
    with pytest.raises(ValueError):
        jacobian_ab_to_mu_nu(np.nan, 1.0)


def test_delta_method_cov_batched_alpha_beta_with_shared_cov_ab():
    """Regression test: batched alpha/beta (J is (n,2,2)) sharing a single
    2x2 cov_ab must NOT use a bare `.T`, which would reverse every axis
    instead of just the last two."""
    alpha = np.array([1.0, 2.0, 3.0])
    beta = np.array([4.0, 5.0, 6.0])
    cov_ab = np.array([[1.3, 0.4], [0.4, 0.9]])

    result = delta_method_cov(cov_ab, alpha, beta)
    assert result.shape == (3, 2, 2)
    for i in range(3):
        J_i = jacobian_ab_to_mu_nu(alpha[i], beta[i])
        expected = J_i @ cov_ab @ J_i.T
        assert np.allclose(result[i], expected)


# ============================================================================
# 5. Parameterization invariance: (alpha,beta) fit vs (mu,nu) fit
# ============================================================================

def test_parameterization_invariance_on_synthetic_curves():
    df = _make_synthetic_df(seed=7, n_curves=8, n_pts=12)
    fits_ab = fitting.fit_all_curves(df, keep_mu_nu=True)
    fits_munu = fitting.fit_all_curves_mu_nu(df)

    report = verification.verify_parameterizations(fits_ab, fits_munu)

    assert len(report) == df["project"].nunique()
    assert (report["alpha_agree"] & report["beta_agree"]).all()
    assert report["max_curve_diff"].max() < 1e-6


def test_fit_all_curves_default_unchanged_by_keep_mu_nu_flag():
    """keep_mu_nu=False must be indistinguishable from the pre-existing
    column set (additive-only guarantee)."""
    df = _make_synthetic_df(seed=3, n_curves=5, n_pts=10)
    fits_default = fitting.fit_all_curves(df)
    fits_explicit_false = fitting.fit_all_curves(df, keep_mu_nu=False)
    pd.testing.assert_frame_equal(fits_default, fits_explicit_false)
    assert "mu" not in fits_default.columns
    assert "nu" not in fits_default.columns


# ============================================================================
# 6. bvn_munu sampling: positivity + mean matches fitted m within MC error
# ============================================================================

def test_bvn_munu_sampling_positive_and_unbiased():
    df = _make_synthetic_df(seed=11, n_curves=10, n_pts=12, cost_type="TPC")
    fits = fitting.fit_all_curves(df, keep_mu_nu=True)
    fits_for_stage2, _ = screening.screen_post_fit(fits)
    dists = distributions.fit_joint_distributions(fits_for_stage2, include_munu=True)

    pred = distributions.PhasingPredictor(dists, model="bvn_munu",
                                          rng=np.random.default_rng(99))
    n = 100_000
    alpha, beta = pred.sample("TPC", n=n)

    assert np.all(alpha > 0)
    assert np.all(beta > 0)

    mu_s, nu_s = ab_to_mu_nu(alpha, beta)
    logit_mu_s = np.log(mu_s / (1.0 - mu_s))
    log_nu_s = np.log(nu_s)

    fitted = next(p for p in dists["bvn_munu"] if p["cost_type"] == "TPC")
    m = np.array(fitted["mu"])

    se_logit_mu = logit_mu_s.std(ddof=1) / np.sqrt(n)
    se_log_nu = log_nu_s.std(ddof=1) / np.sqrt(n)
    assert abs(logit_mu_s.mean() - m[0]) < 6 * se_logit_mu
    assert abs(log_nu_s.mean() - m[1]) < 6 * se_log_nu


def test_phasing_predictor_missing_model_raises():
    df = _make_synthetic_df(seed=2, n_curves=5, n_pts=10)
    fits = fitting.fit_all_curves(df)  # no keep_mu_nu -> no mu/nu columns
    fits_for_stage2, _ = screening.screen_post_fit(fits)
    dists = distributions.fit_joint_distributions(fits_for_stage2)  # no bvn_munu
    with pytest.raises(KeyError):
        distributions.PhasingPredictor(dists, model="bvn_munu")


# ============================================================================
# decompose_correlation: sanity + non-PD flag path
# ============================================================================

def test_decompose_correlation_ab_space_positive_and_flagged_when_appropriate():
    df = _make_synthetic_df(seed=5, n_curves=12, n_pts=12)
    fits = fitting.fit_all_curves(df, keep_mu_nu=True)
    fits_for_stage2, _ = screening.screen_post_fit(fits)
    decomp = correlation.decompose_correlation(fits_for_stage2, space="ab")
    assert set(decomp.columns) >= {
        "cost_type", "space", "n", "observed_r", "estimator_r",
        "population_r", "non_pd_flag", "fallback_r",
    }
    row = decomp.iloc[0]
    if not row["non_pd_flag"]:
        assert -1.0 <= row["population_r"] <= 1.0
    else:
        assert np.isnan(row["population_r"])
        assert row["fallback_r"] == pytest.approx(row["observed_r"])


def test_decompose_correlation_unknown_space_raises():
    df = _make_synthetic_df(seed=5, n_curves=5, n_pts=10)
    fits = fitting.fit_all_curves(df, keep_mu_nu=True)
    fits_for_stage2, _ = screening.screen_post_fit(fits)
    with pytest.raises(ValueError):
        correlation.decompose_correlation(fits_for_stage2, space="bogus")


# ============================================================================
# Incremental-spend input format:
# project, year_from_start, inc_pct_spend, cum_pct_sched (no cost_type)
# ============================================================================

def test_resolve_columns_incremental_format_with_punctuated_headers():
    df = pd.DataFrame({
        "Project": ["P1", "P1"],
        "Year From Start": [0, 1],
        "Inc. Pct. Spend": [10.0, 20.0],
        "Cum. Pct Sched": [0.2, 0.4],
    })
    resolved = data._resolve_columns(df)
    assert resolved["project"] == "Project"
    assert resolved["pct_sched"] == "Cum. Pct Sched"
    assert resolved["inc_pct_cost"] == "Inc. Pct. Spend"
    assert resolved["year"] == "Year From Start"
    assert resolved["cost_type"] is None
    assert resolved["pct_cost"] is None


def test_resolve_columns_prefers_cumulative_when_both_present():
    df = pd.DataFrame({
        "project": ["P1"], "cum_pct_cost": [0.5], "inc_pct_cost": [0.1],
        "pct_sched": [0.3],
    })
    resolved = data._resolve_columns(df)
    assert resolved["pct_cost"] == "cum_pct_cost"
    assert resolved["inc_pct_cost"] is None


def _synthetic_incremental_csv(tmp_path, seed=321, n_projects=8):
    """Write a project/year_from_start/inc_pct_spend/cum_pct_sched CSV with
    known (mu, nu) = (0.5, 9.0) for load_and_validate round-trip testing."""
    rng = np.random.default_rng(seed)
    mu_true, nu_true = 0.5, 9.0
    rows = []
    for p in range(n_projects):
        mu_i = np.clip(mu_true + rng.normal(0, 0.03), 0.05, 0.95)
        nu_i = max(nu_true + rng.normal(0, 1.0), 1.0)
        a, b = mu_nu_to_ab(mu_i, nu_i)
        n_years = rng.integers(6, 10)
        t = np.sort(rng.uniform(0.05, 0.98, n_years))
        y_cum = np.clip(np.sort(beta_cdf(t, a, b)), 0.001, 0.999)
        y_inc = np.diff(np.concatenate([[0.0], y_cum]))
        for yr_idx, (tt, inc) in enumerate(zip(t, y_inc)):
            rows.append({
                "Project": f"PROJ-{p:03d}",
                "Year From Start": yr_idx,
                "Inc. Pct. Spend": inc * 100.0,   # percent scale, on purpose
                "Cum. Pct Sched": tt,
            })
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_and_validate_incremental_format(tmp_path, monkeypatch):
    csv_path = _synthetic_incremental_csv(tmp_path)
    monkeypatch.setattr(config, "DATA_PATH", csv_path)

    df, cols = data.load_and_validate()

    assert cols["cost_type"] is None
    assert cols["pct_cost"] is None
    assert cols["inc_pct_cost"] == "Inc. Pct. Spend"
    assert set(df["cost_type"].unique()) == {config.DEFAULT_COST_TYPE}
    assert df["pct_cost"].max() <= 1.0 + 1e-9
    assert df["pct_cost"].min() >= 0.0
    # cumulative pct_cost must be non-decreasing within each curve
    for _, g in df.groupby(["project", "cost_type"]):
        assert np.all(np.diff(g["pct_cost"].values) >= -1e-9)

    fits = fitting.fit_all_curves(df)
    fitted = fits[fits["status"] == "fit"]
    # recovered alpha, beta should cluster near the true mu=0.5, nu=9 ->
    # alpha = beta = 4.5
    assert fitted["alpha"].mean() == pytest.approx(4.5, rel=0.25)
    assert fitted["beta"].mean() == pytest.approx(4.5, rel=0.25)
    assert (fitted["r2"] > 0.99).all()


# ============================================================================
# CLI: --model bvn_munu without --param munu/both must fail cleanly
# ============================================================================

def test_cli_rejects_bvn_munu_model_without_munu_param(monkeypatch):
    from phasing.pipeline import main as pipeline_main

    monkeypatch.setattr("sys.argv", ["phasing_analysis.py", "--model", "bvn_munu"])
    with pytest.raises(SystemExit) as exc_info:
        pipeline_main()
    assert exc_info.value.code == 2  # argparse.error() exit code, not a KeyError


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
