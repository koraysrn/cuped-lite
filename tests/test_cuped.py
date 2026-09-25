"""Integration-style unit tests for the CUPED class."""

import numpy as np
import pytest

from cuped_lite import CUPED, CUPEDResult


def make_data(rho=0.8, effect=0.2, n=4000, seed=42):
    """Generate a controlled dataset where Corr(X, Y) ~ rho."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=n)
    treatment = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    Y = 10.0 + rho * X + np.sqrt(1 - rho**2) * noise + effect * treatment
    return X, Y, treatment


class TestFit:
    def test_fit_returns_self(self):
        X, Y, t = make_data()
        cuped = CUPED()
        assert cuped.fit(X, Y, t) is cuped

    def test_theta_close_to_rho(self):
        rho = 0.8
        X, Y, t = make_data(rho=rho)
        cuped = CUPED().fit(X, Y, t)
        assert cuped.theta_ == pytest.approx(rho, abs=0.05)

    def test_result_type(self):
        X, Y, t = make_data()
        result = CUPED().fit(X, Y, t).result_
        assert isinstance(result, CUPEDResult)

    def test_variance_reduction_close_to_rho_squared(self):
        rho = 0.7
        X, Y, t = make_data(rho=rho)
        result = CUPED().fit(X, Y, t).result_
        assert result.variance_reduction == pytest.approx(rho**2, abs=0.05)
        assert 0.0 < result.variance_reduction < 1.0

    def test_adjusted_difference_preserves_effect(self):
        X, Y, t = make_data(effect=0.2)
        result = CUPED().fit(X, Y, t).result_
        assert result.adjusted_difference == pytest.approx(0.2, abs=0.05)

    def test_p_value_small_for_large_effect(self):
        X, Y, t = make_data(effect=0.5, n=4000)
        result = CUPED().fit(X, Y, t).result_
        assert result.p_value < 1e-6

    def test_ci_contains_adjusted_difference(self):
        X, Y, t = make_data()
        result = CUPED().fit(X, Y, t).result_
        assert result.ci_lower <= result.adjusted_difference <= result.ci_upper

    def test_relative_lift_percent(self):
        X, Y, t = make_data(effect=0.2)
        result = CUPED().fit(X, Y, t).result_
        # Control adjusted mean ~ 10, effect ~ 0.2 -> ~2% lift.
        assert result.relative_lift == pytest.approx(2.0, abs=0.5)


class TestStrategies:
    def test_per_group_theta_dict(self):
        X, Y, t = make_data(rho=0.8)
        cuped = CUPED(strategy="per_group").fit(X, Y, t)
        assert set(cuped.theta_).issubset({0, 1})
        assert cuped.result_.theta_strategy == "per_group"

    def test_per_group_transform_requires_treatment(self):
        X, Y, t = make_data()
        cuped = CUPED(strategy="per_group").fit(X, Y, t)
        with pytest.raises(ValueError, match="treatment"):
            cuped.transform(X, Y)


class TestTransform:
    def test_transform_matches_manual(self):
        rho = 0.8
        X, Y, t = make_data(rho=rho)
        cuped = CUPED().fit(X, Y, t)
        expected = Y - cuped.theta_ * (X - float(np.mean(X)))
        np.testing.assert_allclose(cuped.transform(X, Y), expected)

    def test_transform_before_fit_raises(self):
        X, Y, _ = make_data()
        with pytest.raises(RuntimeError, match="fit"):
            CUPED().transform(X, Y)

    def test_transform_accepts_new_data(self):
        X, Y, t = make_data()
        cuped = CUPED().fit(X, Y, t)
        X_new, Y_new, _ = make_data(seed=7)
        out = cuped.transform(X_new, Y_new)
        assert out.shape == Y_new.shape
        assert out.dtype == np.float64


class TestWarnings:
    def test_low_correlation_warns(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=400)
        Y = rng.normal(size=400)
        t = rng.integers(0, 2, size=400)
        with pytest.warns(UserWarning, match="low"):
            CUPED().fit(X, Y, t)

    def test_constant_x_warns_and_theta_zero(self):
        rng = np.random.default_rng(0)
        X = np.ones(400)
        Y = rng.normal(size=400)
        t = rng.integers(0, 2, size=400)
        cuped = CUPED()
        with pytest.warns(UserWarning, match="near zero"):
            cuped.fit(X, Y, t)
        assert cuped.theta_ == 0.0

    def test_warnings_recorded_in_result(self):
        rng = np.random.default_rng(0)
        X = np.ones(400)
        Y = rng.normal(size=400)
        t = rng.integers(0, 2, size=400)
        cuped = CUPED()
        with pytest.warns(UserWarning):
            cuped.fit(X, Y, t)
        assert any("near zero" in w for w in cuped.result_.warnings)


class TestThetaCorrection:
    def test_correction_widens_confidence_interval(self):
        rng = np.random.default_rng(5)
        n = 20
        X = rng.normal(size=n)
        t = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
        Y = 10.0 + 0.7 * X + 0.5 * t + rng.normal(scale=0.7, size=n)

        corrected = CUPED(theta_correction=True).fit(X, Y, t).result_
        uncorrected = CUPED(theta_correction=False).fit(X, Y, t).result_
        width_c = corrected.ci_upper - corrected.ci_lower
        width_u = uncorrected.ci_upper - uncorrected.ci_lower
        assert width_c > width_u


class TestSummary:
    def test_summary_string(self):
        X, Y, t = make_data()
        text = CUPED().fit(X, Y, t).summary()
        assert "CUPED Analysis Summary" in text
        assert "p-value" in text

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            CUPED().summary()


class TestRequiredSampleSize:
    def test_cuped_reduces_sample_size(self):
        X, Y, t = make_data(rho=0.8, effect=0.2, n=4000)
        result = CUPED().fit(X, Y, t).required_sample_size(power=0.8)
        assert result.n_adjusted < result.n_original
        assert result.n_saved == result.n_original - result.n_adjusted
        assert result.savings_percent > 0

    def test_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            CUPED().required_sample_size()


class TestGetParams:
    def test_get_params(self):
        cuped = CUPED(strategy="per_group", alpha=0.01, theta_correction=False)
        params = cuped.get_params()
        assert params == {
            "strategy": "per_group",
            "alpha": 0.01,
            "theta_correction": False,
        }
