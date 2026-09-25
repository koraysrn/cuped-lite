"""Public :class:`CUPED` estimator."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import Theta, adjust, compute_theta, sample_correlation, variance
from ._hypothesis import (
    check_leakage,
    cohens_d,
    confidence_interval,
    relative_lift,
    two_sided_pvalue,
    welch_df,
)
from ._power import SampleSizeResult
from ._power import required_sample_size as _required_sample_size
from .result import CUPEDResult
from .validation import EPSILON, validate_fit_inputs, validate_transform_inputs

__all__ = ["CUPED"]


def _capture_warnings(warnings_list: list[str], fn: Callable[[], Any]) -> Any:
    """Run ``fn``, record the ``UserWarning`` messages it emits and re-emit them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    for item in caught:
        message = str(item.message)
        warnings_list.append(message)
        warnings.warn(message, item.category, stacklevel=3)
    return result


class CUPED:
    """CUPED (Controlled Pre-Experiment Data) variance reducer.

    Parameters
    ----------
    strategy:
        ``"pooled"`` estimates a single theta on all data; ``"per_group"``
        estimates one theta per treatment group.
    alpha:
        Significance level used for the p-value decision and confidence
        interval width.
    theta_correction:
        If ``True``, inflate each group's adjusted variance by
        ``1 + 1/(n - 2)`` to account for the uncertainty of the estimated
        theta (Deng et al., 2013).
    """

    def __init__(
        self,
        strategy: str = "pooled",
        alpha: float = 0.05,
        theta_correction: bool = True,
    ) -> None:
        if strategy not in ("pooled", "per_group"):
            raise ValueError("`strategy` must be 'pooled' or 'per_group'.")
        if not 0 < alpha < 1:
            raise ValueError("`alpha` must be between 0 and 1.")
        self.strategy = strategy
        self.alpha = alpha
        self.theta_correction = bool(theta_correction)

        self.theta_: Theta | None = None
        self.rho_: float | None = None
        self.x_mean_: float | None = None
        self.result_: CUPEDResult | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters (scikit-learn-style signature)."""
        return {
            "strategy": self.strategy,
            "alpha": self.alpha,
            "theta_correction": self.theta_correction,
        }

    def fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        treatment: ArrayLike,
    ) -> CUPED:
        """Fit CUPED on the pre-experiment covariate ``X``, the experiment
        metric ``Y`` and the binary group flags ``treatment`` (0/1).

        Returns ``self`` so calls can be chained.
        """
        X_arr, Y_arr, treatment_arr = validate_fit_inputs(X, Y, treatment)

        warnings_list: list[str] = []

        # Data-leakage check: X should not be affected by the treatment.
        _capture_warnings(
            warnings_list,
            lambda: check_leakage(X_arr, treatment_arr, alpha=self.alpha),
        )

        # Estimate theta (with Var(X) ~ 0 guard).
        self.theta_ = _capture_warnings(
            warnings_list,
            lambda: compute_theta(X_arr, Y_arr, strategy=self.strategy, treatment=treatment_arr),
        )

        self.rho_ = sample_correlation(X_arr, Y_arr)
        if abs(self.rho_) < 0.1:
            message = (
                "Correlation between X and Y is low (|rho| < 0.1); CUPED is "
                "unlikely to provide meaningful variance reduction."
            )
            warnings_list.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)

        self.x_mean_ = float(np.mean(X_arr))
        y_adj = self._adjust_array(X_arr, Y_arr, treatment_arr)

        mask_control = treatment_arr == 0
        mask_treatment = treatment_arr == 1
        yc = Y_arr[mask_control]
        yt = Y_arr[mask_treatment]
        yc_adj = y_adj[mask_control]
        yt_adj = y_adj[mask_treatment]
        n_c = int(mask_control.sum())
        n_t = int(mask_treatment.sum())

        mean_c_orig = float(np.mean(yc))
        mean_t_orig = float(np.mean(yt))
        mean_c_adj = float(np.mean(yc_adj))
        mean_t_adj = float(np.mean(yt_adj))
        var_c_orig = variance(yc)
        var_t_orig = variance(yt)
        var_c_adj = variance(yc_adj)
        var_t_adj = variance(yt_adj)

        orig_diff = mean_t_orig - mean_c_orig
        adj_diff = mean_t_adj - mean_c_adj

        if self.theta_correction:
            corr_c = 1.0 + 1.0 / (n_c - 2) if n_c > 2 else 1.0
            corr_t = 1.0 + 1.0 / (n_t - 2) if n_t > 2 else 1.0
        else:
            corr_c = corr_t = 1.0

        se = float(np.sqrt(var_c_adj * corr_c / n_c + var_t_adj * corr_t / n_t))
        df = welch_df(var_c_adj * corr_c, var_t_adj * corr_t, n_c, n_t)
        t_stat = adj_diff / se
        p_value = two_sided_pvalue(t_stat, df)
        ci_lower, ci_upper = confidence_interval(adj_diff, se, df, self.alpha)

        var_orig_pooled = variance(Y_arr)
        var_adj_pooled = variance(y_adj)
        var_reduction = 1.0 - var_adj_pooled / var_orig_pooled if var_orig_pooled > EPSILON else 0.0

        effect_size = cohens_d(mean_c_adj, mean_t_adj, var_c_adj, var_t_adj, n_c, n_t)
        rel_lift = relative_lift(adj_diff, mean_c_adj)

        self.result_ = CUPEDResult(
            theta=self.theta_,
            theta_strategy=self.strategy,
            correlation=self.rho_,
            variance_reduction=var_reduction,
            control_mean_original=mean_c_orig,
            treatment_mean_original=mean_t_orig,
            control_mean_adjusted=mean_c_adj,
            treatment_mean_adjusted=mean_t_adj,
            control_variance_original=var_c_orig,
            treatment_variance_original=var_t_orig,
            control_variance_adjusted=var_c_adj,
            treatment_variance_adjusted=var_t_adj,
            original_difference=orig_diff,
            adjusted_difference=adj_diff,
            effect_size=effect_size,
            relative_lift=rel_lift,
            p_value=p_value,
            t_statistic=t_stat,
            degrees_of_freedom=df,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha=self.alpha,
            n_control=n_c,
            n_treatment=n_t,
            theta_correction=self.theta_correction,
            warnings=warnings_list,
        )
        return self

    def transform(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        treatment: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Apply the fitted theta to new data and return the adjusted metric.

        For ``strategy="per_group"`` a ``treatment`` vector is required.
        """
        if self.theta_ is None or self.x_mean_ is None:
            raise RuntimeError("`CUPED.transform` requires a fitted estimator; call `fit` first.")
        X_arr, Y_arr, treatment_arr = validate_transform_inputs(X, Y, treatment)

        if self.strategy == "pooled":
            assert isinstance(self.theta_, float)
            return adjust(Y_arr, X_arr, self.theta_, self.x_mean_)

        if treatment_arr is None:
            raise ValueError("`strategy='per_group'` requires a `treatment` vector in `transform`.")
        assert isinstance(self.theta_, dict)
        y_adj = np.empty_like(Y_arr, dtype=np.float64)
        for group in (0, 1):
            mask = treatment_arr == group
            x_mean_g = float(np.mean(X_arr[mask]))
            y_adj[mask] = adjust(Y_arr[mask], X_arr[mask], self.theta_[group], x_mean_g)
        return y_adj

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the fitted analysis."""
        if self.result_ is None:
            raise RuntimeError("`CUPED.summary` requires a fitted estimator; call `fit` first.")
        return self.result_.to_text()

    def required_sample_size(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        ratio: float = 1.0,
    ) -> SampleSizeResult:
        """Compare required sample size with and without CUPED.

        Uses the fitted adjusted effect and the pooled standard deviation of
        the original vs. adjusted metric.
        """
        if self.result_ is None:
            raise RuntimeError(
                "`CUPED.required_sample_size` requires a fitted estimator; call `fit` first."
            )
        result = self.result_
        pooled_std_orig = float(
            np.sqrt((result.control_variance_original + result.treatment_variance_original) / 2.0)
        )
        pooled_std_adj = float(
            np.sqrt((result.control_variance_adjusted + result.treatment_variance_adjusted) / 2.0)
        )
        diff = abs(result.adjusted_difference)

        n_orig = _required_sample_size(
            diff, pooled_std_orig, 0.0, alpha=alpha, power=power, ratio=ratio
        )
        n_adj = _required_sample_size(
            diff, pooled_std_adj, 0.0, alpha=alpha, power=power, ratio=ratio
        )
        saved = n_orig - n_adj
        savings_percent = saved / n_orig * 100.0 if n_orig else 0.0

        return SampleSizeResult(
            n_original=n_orig,
            n_adjusted=n_adj,
            n_saved=saved,
            savings_percent=savings_percent,
            power=power,
            alpha=alpha,
            mean_diff=diff,
            pooled_std=pooled_std_orig,
            variance_reduction=result.variance_reduction,
        )

    def _adjust_array(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        treatment: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        assert self.theta_ is not None
        assert self.x_mean_ is not None
        if self.strategy == "pooled":
            assert isinstance(self.theta_, float)
            return adjust(Y, X, self.theta_, self.x_mean_)

        assert isinstance(self.theta_, dict)
        y_adj = np.empty_like(Y, dtype=np.float64)
        for group in (0, 1):
            mask = treatment == group
            x_mean_g = float(np.mean(X[mask]))
            y_adj[mask] = adjust(Y[mask], X[mask], self.theta_[group], x_mean_g)
        return y_adj
