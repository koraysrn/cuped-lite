# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-25

### Added

- `CUPED` class with `fit()`, `transform()`, `summary()`, `get_params()` and
  `required_sample_size()` methods.
- `strategy="pooled"` and `strategy="per_group"` theta estimation strategies.
- `CUPEDResult` structured result object: original/adjusted means, variances,
  variance reduction ratio, p-value, confidence intervals, effect size
  (Cohen's d) and relative lift.
- Welch t-test and Welch-Satterthwaite degrees-of-freedom based confidence
  intervals, with an optional correction for theta estimation error
  (`theta_correction`).
- Power / sample size module: `required_sample_size()` and
  `CUPED.required_sample_size()` for sample-size savings reporting.
- Strict input validation (equal lengths, NaN/inf rejection, numeric type
  checks, exactly two treatment classes, minimum observations per group) and a
  soft warning hierarchy (low rho, near-zero Var(X), suspected data leakage).
- `pytest` unit test suite covering synthetic and known variance-reduction
  scenarios.
- GitHub Actions CI, ruff lint, black format and mypy type-check configuration.
