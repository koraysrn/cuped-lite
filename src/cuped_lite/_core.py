"""Statistical core: covariance, variance, correlation and theta estimation."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from .validation import EPSILON

__all__ = [
    "Theta",
    "adjust",
    "compute_theta",
    "covariance",
    "sample_correlation",
    "variance",
]

#: Estimated theta is either a single pooled float or a per-group dict.
Theta = float | dict[int, float]


def variance(values: NDArray[np.float64], ddof: int = 1) -> float:
    """Return the sample variance of ``values`` (unbiased, ddof=1)."""
    if values.size < 2:
        return float("nan")
    return float(np.var(values, ddof=ddof))


def covariance(x: NDArray[np.float64], y: NDArray[np.float64], ddof: int = 1) -> float:
    """Return the sample covariance between ``x`` and ``y`` (unbiased, ddof=1)."""
    if x.size < 2:
        return float("nan")
    return float(np.cov(x, y, ddof=ddof)[0, 1])


def sample_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Return the Pearson correlation between ``x`` and ``y``.

    Returns 0.0 if either vector has (near-)zero variance.
    """
    vx = variance(x)
    vy = variance(y)
    if vx < EPSILON or vy < EPSILON:
        return 0.0
    return covariance(x, y) / np.sqrt(vx * vy)


def compute_theta(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    strategy: str = "pooled",
    treatment: NDArray[np.int64] | None = None,
) -> Theta:
    """Estimate the CUPED coefficient ``theta = Cov(Y, X) / Var(X)``.

    ``strategy="pooled"`` returns a single float estimated on all data.
    ``strategy="per_group"`` returns a ``{group: theta}`` dict; ``treatment``
    is required in that case.

    If ``Var(X)`` is (near) zero a soft warning is emitted and theta is set
    to 0.0 to avoid a division by zero.
    """
    if strategy == "pooled":
        var_x = variance(X)
        if var_x < EPSILON:
            warnings.warn(
                "Var(X) is near zero; the covariate is constant, theta is set to 0.",
                UserWarning,
                stacklevel=2,
            )
            return 0.0
        return covariance(X, Y) / var_x

    if strategy == "per_group":
        if treatment is None:
            raise ValueError("strategy='per_group' requires a `treatment` vector.")
        thetas: dict[int, float] = {}
        for group in (0, 1):
            mask = treatment == group
            xg = X[mask]
            yg = Y[mask]
            var_xg = variance(xg)
            if var_xg < EPSILON:
                warnings.warn(
                    f"Var(X) is near zero in group {group}; theta is set to 0.",
                    UserWarning,
                    stacklevel=2,
                )
                thetas[group] = 0.0
            else:
                thetas[group] = covariance(xg, yg) / var_xg
        return thetas

    raise ValueError(f"Unknown strategy {strategy!r}; expected 'pooled' or 'per_group'.")


def adjust(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    theta: float,
    x_mean: float,
) -> NDArray[np.float64]:
    """Return the CUPED-adjusted metric ``Y - theta * (X - x_mean)``."""
    return Y - theta * (X - x_mean)
