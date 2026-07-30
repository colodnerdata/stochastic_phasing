"""STAGE 5 — JOINT DISTRIBUTIONS + PREDICTOR.

`fit_joint_distributions` and `PhasingPredictor` operate on
`fits_for_stage2` — bound-contact fits have already been excluded by
`phasing.screening.screen_post_fit`, so they don't distort the fitted
BVN parameters used to generate forecasts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .fitting import beta_cdf


def fit_joint_distributions(fits_for_stage2: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("STAGE 5: JOINT DISTRIBUTION FITS")
    print("=" * 70)

    dists = {"bvn_unit": [], "bvn_log": []}
    for ct, sub in fits_for_stage2.groupby("cost_type"):
        data_u = sub[["alpha", "beta"]].values
        data_l = np.log(data_u)
        for key, d in (("bvn_unit", data_u), ("bvn_log", data_l)):
            mean = d.mean(axis=0)
            cov = np.cov(d.T)
            sd = np.sqrt(np.diag(cov))
            rho = cov[0, 1] / (sd[0] * sd[1]) if sd.prod() > 0 else 0.0
            dists[key].append(dict(
                cost_type=ct, n=int(len(sub)),
                mu=[float(mean[0]), float(mean[1])],
                cov=[[float(cov[0, 0]), float(cov[0, 1])],
                     [float(cov[1, 0]), float(cov[1, 1])]],
                sigma=[float(sd[0]), float(sd[1])],
                rho=float(rho),
            ))
        u = dists["bvn_unit"][-1]
        l = dists["bvn_log"][-1]
        print(f"\n{ct}:")
        print(f"  Unit BVN: μ=({u['mu'][0]:.3f},{u['mu'][1]:.3f}) "
              f"σ=({u['sigma'][0]:.3f},{u['sigma'][1]:.3f}) ρ={u['rho']:+.3f}")
        print(f"  Log  BVN: μ=({l['mu'][0]:.3f},{l['mu'][1]:.3f}) "
              f"σ=({l['sigma'][0]:.3f},{l['sigma'][1]:.3f}) ρ={l['rho']:+.3f}")

    with open(config.OUTPUT_DIR / config.DISTRIBUTIONS_FILENAME, "w") as f:
        json.dump(dists, f, indent=2)
    print(f"\nSaved -> {config.OUTPUT_DIR / config.DISTRIBUTIONS_FILENAME}")
    return dists


class PhasingPredictor:
    """
    Samples (α, β) for a future project and evaluates Beta CDF phasing.

    Default model 'bvn_log': (ln α, ln β) ~ BVN  (bivariate lognormal).
    Guarantees positive parameters; captures correlation; right-skew friendly.
    'bvn_unit' samples in unit space with rejection for positivity.
    """

    def __init__(self, dists: dict, model: str = "bvn_log",
                 rng: np.random.Generator | None = None):
        self.model = model
        self.params = {p["cost_type"]: p for p in dists[model]}
        self.rng = rng or np.random.default_rng(config.RNG_SEED)

    def sample(self, cost_type: str, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        p = self.params[cost_type]
        mu, cov = np.array(p["mu"]), np.array(p["cov"])
        if self.model == "bvn_log":
            z = self.rng.multivariate_normal(mu, cov, size=n)
            ab = np.exp(z)
        else:  # bvn_unit with rejection for positivity
            out = []
            need = n
            while need > 0:
                z = self.rng.multivariate_normal(mu, cov, size=max(need * 2, 64))
                z = z[(z > 0.05).all(axis=1)]
                out.append(z[:need])
                need = n - sum(len(o) for o in out)
            ab = np.vstack(out)
        return ab[:, 0], ab[:, 1]

    def predict(self, cost_type: str, duration_months: int,
                n_samples: int = config.N_PRED_SAMPLES):
        """Monthly cumulative phasing samples over a given duration.
        Returns (months 0..D, samples of shape (n_samples, D+1))."""
        months = np.arange(0, duration_months + 1)
        t = months / duration_months
        a, b = self.sample(cost_type, n_samples)
        cum = beta_cdf(t[None, :], a[:, None], b[:, None])
        cum[:, 0], cum[:, -1] = 0.0, 1.0
        return months, cum

    def quantile_table(self, cost_type: str, duration_months: int,
                       qs=(0.05, 0.25, 0.50, 0.75, 0.95),
                       n_samples: int = config.N_PRED_SAMPLES) -> pd.DataFrame:
        months, cum = self.predict(cost_type, duration_months, n_samples)
        out = {"month": months}
        for q in qs:
            out[f"q{int(q * 100):02d}"] = np.quantile(cum, q, axis=0)
        out["mean"] = cum.mean(axis=0)
        return pd.DataFrame(out)


def plot_prediction(pred: PhasingPredictor, cost_type: str,
                    duration_months: int):
    months, cum = pred.predict(cost_type, duration_months)
    q05, q25, q50, q75, q95 = (np.quantile(cum, q, axis=0)
                               for q in (0.05, 0.25, 0.50, 0.75, 0.95))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: cumulative fan chart
    ax = axes[0]
    col = config.COLORS.get(cost_type, "#1f3864")
    ax.fill_between(months, q05, q95, alpha=0.2, color=col, label="90% interval")
    ax.fill_between(months, q25, q75, alpha=0.35, color=col, label="50% interval")
    ax.plot(months, q50, color=col, lw=2.5, label="Median")
    # a few spaghetti draws for texture
    for k in range(12):
        ax.plot(months, cum[k], color="gray", alpha=0.3, lw=0.8)
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative % of Total Cost")
    ax.set_title(f"{cost_type} Cumulative Phasing — {duration_months}-month project",
                 fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xlim(0, duration_months); ax.set_ylim(0, 1.02)

    # Panel 2: incremental (monthly) spend distribution
    ax = axes[1]
    inc = np.diff(cum, axis=1)
    i05, i50, i95 = (np.quantile(inc, q, axis=0) for q in (0.05, 0.50, 0.95))
    mid = months[1:]
    ax.fill_between(mid, i05, i95, alpha=0.2, color=col, label="90% interval")
    ax.plot(mid, i50, color=col, lw=2.5, label="Median")
    ax.set_xlabel("Month"); ax.set_ylabel("Monthly % of Total Cost")
    ax.set_title("Implied Monthly Spend Profile", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, duration_months)

    fig.suptitle(f"Stochastic Phasing Forecast ({pred.model})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.OUTPUT_DIR / "05_tpc_prediction.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
