"""Power analysis and sample-size estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

__all__ = ["SampleSizeResult", "required_sample_size"]


@dataclass(slots=True)
class SampleSizeResult:
    """Outcome of a sample-size comparison (with vs. without CUPED)."""

    n_original: int
    n_adjusted: int
    n_saved: int
    savings_percent: float
    power: float
    alpha: float
    mean_diff: float
    pooled_std: float
    variance_reduction: float

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return asdict(self)


def required_sample_size(
    mean_diff: float,
    pooled_std: float,
    variance_reduction: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> int:
    """Return the minimum per-group sample size (control group).

    Two-sample test with equal variance ``pooled_std**2`` and allocation
    ``ratio = n_treatment / n_control``:

        n_control = (z_alpha + z_beta)^2 * (sigma^2 + sigma^2 / ratio) / delta^2

    where ``sigma^2 = pooled_std**2 * (1 - variance_reduction)`` reflects the
    CUPED-adjusted variance.
    """
    effect = abs(mean_diff)
    if effect <= 0 or pooled_std <= 0:
        raise ValueError("`mean_diff` and `pooled_std` must be non-zero and positive.")
    if not 0 < alpha < 1:
        raise ValueError("`alpha` must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("`power` must be between 0 and 1.")
    if ratio <= 0:
        raise ValueError("`ratio` must be positive.")
    if not 0 <= variance_reduction < 1:
        raise ValueError("`variance_reduction` must be in [0, 1).")

    z_alpha = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_beta = stats.norm.ppf(power)
    sigma2 = pooled_std**2 * (1.0 - variance_reduction)
    n_control = (z_alpha + z_beta) ** 2 * (sigma2 + sigma2 / ratio) / (effect**2)
    return int(np.ceil(n_control))
