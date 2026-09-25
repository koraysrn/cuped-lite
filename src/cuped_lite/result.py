"""Structured result object returned by CUPED analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ._core import Theta

__all__ = ["CUPEDResult"]


@dataclass(slots=True)
class CUPEDResult:
    """Structured outcome of a :class:`CUPED` analysis.

    All means, variances, the p-value, confidence intervals, effect size and
    relative lift are directly readable as attributes.
    """

    theta: Theta
    theta_strategy: str
    correlation: float
    variance_reduction: float

    control_mean_original: float
    treatment_mean_original: float
    control_mean_adjusted: float
    treatment_mean_adjusted: float

    control_variance_original: float
    treatment_variance_original: float
    control_variance_adjusted: float
    treatment_variance_adjusted: float

    original_difference: float
    adjusted_difference: float
    effect_size: float
    relative_lift: float

    p_value: float
    t_statistic: float
    degrees_of_freedom: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    alpha: float

    n_control: int
    n_treatment: int
    theta_correction: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        """Total number of observations."""
        return self.n_control + self.n_treatment

    @property
    def absolute_effect(self) -> float:
        """Adjusted absolute effect (treatment minus control)."""
        return self.adjusted_difference

    @property
    def variance_reduction_percent(self) -> float:
        """Variance reduction expressed as a percentage."""
        return self.variance_reduction * 100.0

    @property
    def significant(self) -> bool:
        """Whether the result is significant at the configured ``alpha``."""
        return self.p_value < self.alpha

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary (properties included)."""
        data = asdict(self)
        data["n_total"] = self.n_total
        data["absolute_effect"] = self.absolute_effect
        data["variance_reduction_percent"] = self.variance_reduction_percent
        data["significant"] = self.significant
        return data

    def _format_theta(self) -> str:
        if isinstance(self.theta, dict):
            parts = [f"group {k}: {v:.4f}" for k, v in sorted(self.theta.items())]
            return ", ".join(parts)
        return f"{self.theta:.4f}"

    def to_text(self) -> str:
        """Return a human-readable multi-line summary string."""
        theta = self._format_theta()
        header = [
            "CUPED Analysis Summary",
            "======================",
            f"Theta ({self.theta_strategy})          : {theta}",
            f"Correlation (X, Y)      : {self.correlation:.6f}",
            f"Variance reduction      : {self.variance_reduction_percent:.2f}%",
            "",
            "Metric means",
            "------------",
            f"  Original mean           {self.control_mean_original:>12.4f}   "
            f"{self.treatment_mean_original:>12.4f}",
            f"  Adjusted mean           {self.control_mean_adjusted:>12.4f}   "
            f"{self.treatment_mean_adjusted:>12.4f}",
            "",
            "Variances",
            "---------",
            f"  Original variance       {self.control_variance_original:>12.4f}   "
            f"{self.treatment_variance_original:>12.4f}",
            f"  Adjusted variance       {self.control_variance_adjusted:>12.4f}   "
            f"{self.treatment_variance_adjusted:>12.4f}",
            "",
            "Hypothesis test (Welch, two-sided)",
            "-----------------------------------",
            f"  Adjusted difference     : {self.adjusted_difference:.6f}",
            f"  Standard error          : {self.standard_error:.6f}",
            f"  t-statistic             : {self.t_statistic:.6f}",
            f"  Degrees of freedom      : {self.degrees_of_freedom:.2f}",
            f"  p-value                 : {self.p_value:.6g}",
            f"  {int((1 - self.alpha) * 100)}% CI              : "
            f"[{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
            f"  Cohen's d               : {self.effect_size:.6f}",
            f"  Relative lift           : {self.relative_lift:.4f}%",
            "",
            "Sample sizes",
            "------------",
            f"  Control: {self.n_control}   Treatment: {self.n_treatment}   Total: {self.n_total}",
        ]
        return "\n".join(header)

    def __repr__(self) -> str:
        return (
            f"CUPEDResult(theta={self.theta!r}, variance_reduction="
            f"{self.variance_reduction:.4f}, p_value={self.p_value:.6g}, "
            f"effect_size={self.effect_size:.4f}, "
            f"relative_lift={self.relative_lift:.4f}%)"
        )
