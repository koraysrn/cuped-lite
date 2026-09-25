"""Hypothesis-testing layer built on top of :mod:`scipy.stats`."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ._core import variance
from .validation import EPSILON

__all__ = [
    "check_leakage",
    "cohens_d",
    "confidence_interval",
    "relative_lift",
    "two_sided_pvalue",
    "welch_df",
    "welch_ttest",
]


def welch_df(var_control: float, var_variant: float, n_control: int, n_variant: int) -> float:
    """Return the Welch-Satterthwaite degrees of freedom."""
    num = (var_control / n_control + var_variant / n_variant) ** 2
    den = (var_control / n_control) ** 2 / (n_control - 1) + (var_variant / n_variant) ** 2 / (
        n_variant - 1
    )
    return float(num / den)


def two_sided_pvalue(t_statistic: float, df: float) -> float:
    """Return the two-sided p-value for a Student-t statistic."""
    return float(2 * stats.t.sf(abs(t_statistic), df))


def welch_ttest(
    control: NDArray[np.float64], variant: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    """Run a two-sample Welch t-test.

    Returns ``(t_statistic, p_value, degrees_of_freedom, mean_difference)``
    where ``mean_difference = mean(variant) - mean(control)``.
    """
    n_c = control.size
    n_v = variant.size
    var_c = float(np.var(control, ddof=1))
    var_v = float(np.var(variant, ddof=1))
    se = np.sqrt(var_c / n_c + var_v / n_v)
    diff = float(np.mean(variant) - np.mean(control))
    t_stat = diff / se
    df = welch_df(var_c, var_v, n_c, n_v)
    p_value = two_sided_pvalue(t_stat, df)
    return t_stat, p_value, df, diff


def confidence_interval(
    diff: float, se: float, df: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Return the two-sided ``(1 - alpha)`` confidence interval for ``diff``."""
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    return diff - t_crit * se, diff + t_crit * se


def cohens_d(
    mean_control: float,
    mean_variant: float,
    var_control: float,
    var_variant: float,
    n_control: int,
    n_variant: int,
) -> float:
    """Return Cohen's d using the sample-size weighted pooled standard deviation."""
    numerator = (n_control - 1) * var_control + (n_variant - 1) * var_variant
    denominator = n_control + n_variant - 2
    pooled_std = np.sqrt(numerator / denominator)
    if pooled_std < EPSILON:
        return 0.0
    return float((mean_variant - mean_control) / pooled_std)


def relative_lift(diff: float, control_mean: float) -> float:
    """Return the relative lift in percent: ``diff / control_mean * 100``."""
    if abs(control_mean) < EPSILON:
        return 0.0
    return float(diff / control_mean * 100.0)


def check_leakage(
    X: NDArray[np.float64], treatment: NDArray[np.int64], alpha: float = 0.05
) -> bool:
    """Warn if the covariate ``X`` differs significantly between groups.

    A significant difference suggests that ``X`` may have been measured after
    the treatment and therefore be affected by it (data leakage). Returns
    ``True`` when leakage is suspected.
    """
    x_control = X[treatment == 0]
    x_variant = X[treatment == 1]
    if variance(x_control) < EPSILON or variance(x_variant) < EPSILON:
        # A constant covariate cannot reveal leakage; skip the test to avoid
        # numerical precision warnings from scipy.
        return False
    _, p_value = stats.ttest_ind(x_control, x_variant, equal_var=False)
    if p_value < alpha:
        warnings.warn(
            "The covariate X differs significantly between treatment groups; "
            "X may be affected by the treatment (data leakage risk).",
            UserWarning,
            stacklevel=2,
        )
        return True
    return False
