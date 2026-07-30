"""Mean-precision parameterization of the Beta phasing curve.

mu = alpha / (alpha + beta)   in (0, 1)   mean spend timing
nu = alpha + beta             in (0, inf) precision / concentration

This is a smooth bijection (0,inf)^2 -> (0,1)x(0,inf): the fitted curve,
likelihood value, and predictions are identical under either
parameterization. Only conditioning and interpretability change. mu, nu
are NOT Fisher-orthogonal in general -- do not treat them as such.
"""

from __future__ import annotations

import numpy as np


def ab_to_mu_nu(alpha, beta):
    """Map Beta shape parameters (alpha, beta) to mean-precision (mu, nu).

    Parameters
    ----------
    alpha, beta : scalar or array_like, > 0

    Returns
    -------
    mu, nu : same shape as input
        mu = alpha / (alpha + beta) in (0, 1); nu = alpha + beta > 0.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if not (np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta))):
        raise ValueError("ab_to_mu_nu: alpha, beta must be finite")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("ab_to_mu_nu: alpha, beta must be > 0")
    nu = alpha + beta
    mu = alpha / nu
    return mu, nu


def mu_nu_to_ab(mu, nu):
    """Inverse of ab_to_mu_nu: mean-precision (mu, nu) -> (alpha, beta).

    Parameters
    ----------
    mu : scalar or array_like, strictly in (0, 1)
    nu : scalar or array_like, > 0

    Returns
    -------
    alpha, beta : same shape as input
        alpha = mu * nu; beta = (1 - mu) * nu.
    """
    mu = np.asarray(mu, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(nu))):
        raise ValueError("mu_nu_to_ab: mu, nu must be finite")
    if np.any(mu <= 0) or np.any(mu >= 1):
        raise ValueError("mu_nu_to_ab: mu must be strictly in (0, 1)")
    if np.any(nu <= 0):
        raise ValueError("mu_nu_to_ab: nu must be > 0")
    alpha = mu * nu
    beta = (1.0 - mu) * nu
    return alpha, beta


def jacobian_ab_to_mu_nu(alpha, beta):
    """Jacobian d(mu, nu)/d(alpha, beta), evaluated at (alpha, beta).

    Returns
    -------
    J : ndarray, shape (2, 2) for scalar input or (..., 2, 2) for array
        input, with J[..., 0, :] = d(mu)/d(alpha, beta) and
        J[..., 1, :] = d(nu)/d(alpha, beta).
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    s = alpha + beta
    dmu_da = beta / s**2
    dmu_db = -alpha / s**2
    dnu_da = np.ones_like(s)
    dnu_db = np.ones_like(s)
    row_mu = np.stack([dmu_da, dmu_db], axis=-1)
    row_nu = np.stack([dnu_da, dnu_db], axis=-1)
    return np.stack([row_mu, row_nu], axis=-2)


def delta_method_cov(cov_ab, alpha, beta):
    """Delta-method map of Cov(alpha_hat, beta_hat) to Cov(mu_hat, nu_hat).

    Parameters
    ----------
    cov_ab : array_like, shape (2, 2) or (..., 2, 2)
        Covariance of (alpha_hat, beta_hat), e.g. from curve_fit's pcov.
    alpha, beta : scalar or array_like, matching the batch shape of cov_ab

    Returns
    -------
    cov_mu_nu : ndarray, same shape as cov_ab
        Cov(mu_hat, nu_hat) ~= J . cov_ab . J^T.
    """
    cov_ab = np.asarray(cov_ab, dtype=float)
    J = jacobian_ab_to_mu_nu(alpha, beta)
    if cov_ab.ndim == 2:
        return J @ cov_ab @ J.T
    return np.einsum("...ij,...jk,...lk->...il", J, cov_ab, J)
