# cuped-lite


A dependency-light Python library that applies **CUPED** (Controlled
Pre-Experiment Data) variance reduction to digital A/B tests. By exploiting
pre-experiment data, it lowers the variance of the experiment metric so that
the same statistical power is reached with **fewer samples and in less time**.

The library is in-memory only and depends **solely on NumPy and SciPy**.
No pandas, no database drivers, no web UI.

## Features

- Optimal coefficient estimation: `theta = Cov(Y, X) / Var(X)`.
- Adjusted metric construction: `Y_adj = Y - theta * (X - X_mean)`.
- Two theta strategies: `"pooled"` (default) and `"per_group"`.
- Variance reduction ratio reporting.
- Two-sample Welch t-test, two-sided p-value and confidence intervals.
- Effect size (Cohen's d), absolute effect and relative lift.
- Optional correction for theta estimation error in small samples.
- Power / sample-size module answering "how many samples did CUPED save?".
- Strict input validation and a soft warning hierarchy (low rho, near-zero
  `Var(X)`, suspected data leakage).

## Installation

```bash
pip install cuped-lite
```

### From source

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from cuped_lite import CUPED

rng = np.random.default_rng(42)

# Pre-experiment covariate (e.g. historical metric value)
X = rng.normal(10.0, 2.0, size=5000)

# Experiment metric: correlated with X, with a small treatment effect
treatment = rng.integers(0, 2, size=5000)
noise = rng.normal(0.0, 1.0, size=5000)
Y = 5.0 + 0.8 * X + 0.15 * treatment + noise

cuped = CUPED(strategy="pooled", alpha=0.05)
cuped.fit(X, Y, treatment)

print(cuped.summary())
print(cuped.required_sample_size())
```

## API

### `CUPED`

```python
CUPED(strategy="pooled", alpha=0.05, theta_correction=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"pooled"` | `"pooled"` estimates one theta across all data; `"per_group"` estimates one theta per treatment group. |
| `alpha` | `float` | `0.05` | Significance level for confidence intervals and p-values. |
| `theta_correction` | `bool` | `True` | Inflate each group's adjusted variance by `1 + 1/(n - 2)` to account for theta being estimated (Deng et al., 2013). |

Methods:

- `fit(X, Y, treatment) -> CUPED` — validates inputs, estimates theta, computes
  the adjusted metric, runs the hypothesis test and stores a `CUPEDResult`.
  Returns `self` for chaining.
- `transform(X, Y, treatment=None) -> np.ndarray` — applies the fitted theta to
  new data and returns the adjusted metric vector. `treatment` is required when
  `strategy="per_group"`.
- `summary() -> str` — returns a human-readable report.
- `required_sample_size(power=0.8, alpha=0.05, ratio=1.0) -> SampleSizeResult`
  — compares the required sample size with and without CUPED.
- `get_params() -> dict` — returns the constructor parameters.

Fitted attributes (available after `fit()`):

- `theta_` — estimated coefficient(s); `float` for pooled, `dict` for per-group.
- `rho_` — correlation between `X` and `Y`.
- `result_` — the `CUPEDResult` object.

### `CUPEDResult`

Structured result object. Key fields:

- `theta`, `theta_strategy`, `correlation`, `variance_reduction`
- `control_mean_original`, `treatment_mean_original`
- `control_mean_adjusted`, `treatment_mean_adjusted`
- `p_value`, `t_statistic`, `degrees_of_freedom`, `standard_error`
- `ci_lower`, `ci_upper`
- `effect_size` (Cohen's d), `relative_lift` (%), `absolute_effect`
- `n_control`, `n_treatment`, `n_total`
- `warnings` — soft warnings emitted during fitting

Methods: `to_dict()` and `to_text()`.

### `required_sample_size`

```python
from cuped_lite import required_sample_size

required_sample_size(
    mean_diff,
    pooled_std,
    variance_reduction=0.0,
    alpha=0.05,
    power=0.8,
    ratio=1.0,
    two_sided=True,
)
```

Returns the minimum per-group sample size for a two-sample test.

## Methodology

Given the pre-experiment metric `X` and the experiment metric `Y`:

1. Estimate `theta = Cov(Y, X) / Var(X)` (pooled) or per group.
2. Build the adjusted metric `Y_adj = Y - theta * (X - X_mean)`.
3. The asymptotic variance reduction is `1 - rho^2`, where `rho` is the
   correlation between `X` and `Y`.
4. Run a two-sample Welch t-test on the adjusted metric.

Confidence intervals use the adjusted variances and the Welch-Satterthwaite
degrees of freedom. With `theta_correction=True`, each group's adjusted
variance is inflated by `1 + 1/(n - 2)` to reflect the uncertainty of the
estimated theta.

## Validation and warnings

Hard errors (`TypeError` / `ValueError`):

- Non-numeric, boolean, object or string input.
- NaN or infinite values.
- Length mismatch between `X`, `Y` and `treatment`.
- `treatment` not containing exactly the two classes `{0, 1}`.
- Fewer than 2 observations per group.

Soft warnings (`warnings.warn`):

- `Var(X) ~ 0` — theta is set to 0.
- `|rho| < 0.1` — CUPED is unlikely to provide meaningful gains.
- Covariate imbalance between groups — possible data leakage (X affected by
  treatment).

## Project structure

```
cuped-lite/
├── src/cuped_lite/
│   ├── __init__.py       # public exports
│   ├── cuped.py          # CUPED estimator (fit/transform/summary/...)
│   ├── validation.py     # strict input validation
│   ├── _core.py          # theta, covariance, variance, correlation
│   ├── _hypothesis.py    # Welch t-test, CI, effect size, leakage check
│   ├── _power.py         # sample-size estimation
│   ├── result.py         # CUPEDResult data object
│   └── py.typed
├── tests/                # pytest unit tests
├── examples/             # end-to-end simulation
├── .github/workflows/    # CI (test, lint, format, type check)
└── pyproject.toml
```

## Development

```bash
pip install -e ".[dev]"

pytest            # run tests
ruff check .      # lint
black --check .   # format check
mypy src          # type check
python -m build   # build sdist and wheel
```

## License

MIT. See [LICENSE](LICENSE).

## References

- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). *Improving the
  Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment
  Data*. WSDM '13.
