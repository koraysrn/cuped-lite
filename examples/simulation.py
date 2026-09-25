"""End-to-end simulation of cuped-lite, run like a real experimenter would.

Run with::

    python examples/simulation.py

It simulates an e-commerce A/B test and demonstrates variance reduction,
hypothesis testing, effect size/lift reporting, the per-group strategy and the
soft-warning hierarchy.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from cuped_lite import CUPED


def simulate_experiment(n=50_000, rho=0.75, lift=0.02, seed=42):
    """Simulate an e-commerce experiment.

    ``X`` is the historical average order value (pre-experiment covariate),
    ``Y`` is the average order value during the experiment. ``lift`` is the
    true relative treatment effect.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=50.0, scale=20.0, size=n)
    treatment = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 20.0 * np.sqrt(1 - rho**2), size=n)
    effect = 50.0 * lift
    Y = 50.0 + rho * (X - 50.0) + noise + effect * treatment
    return X, Y, treatment


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def raw_analysis(Y: np.ndarray, treatment: np.ndarray) -> None:
    """Standard Welch t-test WITHOUT CUPED (the baseline)."""
    yc = Y[treatment == 0]
    yt = Y[treatment == 1]
    t_stat, p_value = stats.ttest_ind(yt, yc, equal_var=False)
    print(f"Raw Welch t-test          : t = {t_stat:.4f}, p = {p_value:.6g}")
    print(f"Raw variance (pooled)     : {np.var(Y, ddof=1):.4f}")


def main() -> None:
    X, Y, treatment = simulate_experiment()

    section("1. Experiment simulation (n = 50,000)")
    print(f"Pre-experiment covariate X : mean={X.mean():.2f}, std={X.std(ddof=1):.2f}")
    print(f"Experiment metric Y        : mean={Y.mean():.2f}, std={Y.std(ddof=1):.2f}")
    print(f"Control / treatment split  : {(treatment == 0).sum()} / {(treatment == 1).sum()}")

    section("2. Baseline analysis (no CUPED)")
    raw_analysis(Y, treatment)

    section("3. CUPED analysis (pooled strategy)")
    cuped = CUPED(strategy="pooled", alpha=0.05, theta_correction=True).fit(X, Y, treatment)
    print(cuped.summary())

    section("4. Sample-size savings")
    sizes = cuped.required_sample_size(power=0.8, alpha=0.05)
    print(f"Required per group WITHOUT CUPED : {sizes.n_original:,}")
    print(f"Required per group WITH CUPED    : {sizes.n_adjusted:,}")
    print(f"Samples saved (per group)        : {sizes.n_saved:,}")
    print(f"Savings percentage               : {sizes.savings_percent:.1f}%")

    section("5. Per-group strategy")
    cuped_pg = CUPED(strategy="per_group").fit(X, Y, treatment)
    print(f"Theta per group : {cuped_pg.theta_}")
    print(f"Adjusted difference (per_group) : {cuped_pg.result_.adjusted_difference:.4f}")
    print(f"Adjusted difference (pooled)    : {cuped.result_.adjusted_difference:.4f}")

    section("6. transform() on a holdout slice")
    holdout = np.arange(10)
    y_adj = cuped.transform(X[holdout], Y[holdout])
    print("First 5 raw Y       :", np.round(Y[holdout[:5]], 2))
    print("First 5 adjusted Y  :", np.round(y_adj[:5], 2))

    section("7. Warning hierarchy (soft warnings)")
    rng = np.random.default_rng(0)
    # (a) low-correlation covariate
    print("\n-- Low-correlation covariate --")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CUPED().fit(rng.normal(size=500), rng.normal(size=500), rng.integers(0, 2, 500))
    for w in caught:
        print(f"  [warn] {w.message}")

    # (b) data-leakage suspicion: X differs between groups
    print("\n-- Data-leakage suspicion (X affected by treatment) --")
    X_leak = np.concatenate([rng.normal(0, 1, 300), rng.normal(1.5, 1, 300)])
    t_leak = np.concatenate([np.zeros(300, int), np.ones(300, int)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CUPED().fit(X_leak, rng.normal(size=600), t_leak)
    for w in caught:
        print(f"  [warn] {w.message}")

    section("8. Hard validation errors")
    print("Trying NaN input ...")
    bad = X.copy()
    bad[0] = np.nan
    try:
        CUPED().fit(bad, Y, treatment)
    except ValueError as exc:
        print(f"  [error] {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
