"""Unit tests for the statistical core."""

import numpy as np
import pytest

from cuped_lite._core import (
    adjust,
    compute_theta,
    covariance,
    sample_correlation,
    variance,
)


class TestVarianceCovariance:
    def test_variance_matches_numpy(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert variance(x) == pytest.approx(np.var(x, ddof=1))

    def test_covariance_matches_numpy(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 5.0, 7.0])
        assert covariance(x, y) == pytest.approx(np.cov(x, y, ddof=1)[0, 1])

    def test_correlation_positive_for_linked_variables(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = 0.7 * x + rng.normal(scale=0.5, size=500)
        rho = sample_correlation(x, y)
        assert 0.5 < rho < 1.0

    def test_correlation_zero_when_constant(self):
        x = np.ones(10)
        y = np.arange(10.0)
        assert sample_correlation(x, y) == 0.0


class TestComputeTheta:
    def test_pooled_known_value(self):
        # Y = 2X + 1 exactly -> theta must equal 2.
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = 2.0 * X + 1.0
        assert compute_theta(X, Y, strategy="pooled") == pytest.approx(2.0)

    def test_constant_x_warns_and_returns_zero(self):
        X = np.ones(10)
        Y = np.arange(10.0)
        with pytest.warns(UserWarning, match="near zero"):
            theta = compute_theta(X, Y, strategy="pooled")
        assert theta == 0.0

    def test_per_group(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=200)
        treatment = np.zeros(200, dtype=int)
        treatment[100:] = 1
        Y = 3.0 * X + rng.normal(scale=0.1, size=200)
        thetas = compute_theta(X, Y, strategy="per_group", treatment=treatment)
        assert set(thetas) == {0, 1}
        assert thetas[0] == pytest.approx(3.0, abs=0.05)
        assert thetas[1] == pytest.approx(3.0, abs=0.05)

    def test_per_group_requires_treatment(self):
        with pytest.raises(ValueError, match="treatment"):
            compute_theta(np.ones(10), np.arange(10.0), strategy="per_group")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_theta(np.ones(10), np.arange(10.0), strategy="bogus")


class TestAdjust:
    def test_formula(self):
        X = np.array([1.0, 2.0, 3.0])
        Y = np.array([3.0, 5.0, 7.0])
        theta = 2.0
        x_mean = float(X.mean())
        expected = Y - theta * (X - x_mean)
        np.testing.assert_allclose(adjust(Y, X, theta, x_mean), expected)
