"""Unit tests for the hypothesis-testing layer."""

import numpy as np
import pytest
from scipy import stats

from cuped_lite._hypothesis import (
    check_leakage,
    cohens_d,
    confidence_interval,
    relative_lift,
    welch_ttest,
)


class TestWelchTTest:
    def test_matches_scipy(self):
        rng = np.random.default_rng(0)
        control = rng.normal(10.0, 2.0, 100)
        variant = rng.normal(10.5, 2.0, 100)
        t_stat, p_value, _df, diff = welch_ttest(control, variant)
        res = stats.ttest_ind(variant, control, equal_var=False)
        assert t_stat == pytest.approx(res.statistic)
        assert p_value == pytest.approx(res.pvalue)
        assert diff == pytest.approx(variant.mean() - control.mean())

    def test_detects_large_effect(self):
        rng = np.random.default_rng(1)
        control = rng.normal(10.0, 1.0, 500)
        variant = rng.normal(11.0, 1.0, 500)
        _t_stat, p_value, _df, _diff = welch_ttest(control, variant)
        assert p_value < 1e-3


class TestConfidenceInterval:
    def test_symmetric_around_estimate(self):
        lo, hi = confidence_interval(0.5, 0.1, 50.0, alpha=0.05)
        assert lo < 0.5 < hi
        assert pytest.approx((lo + hi) / 2) == 0.5

    def test_wider_for_smaller_df(self):
        lo_small, hi_small = confidence_interval(0.5, 0.1, 5.0)
        lo_large, hi_large = confidence_interval(0.5, 0.1, 100.0)
        assert (hi_small - lo_small) > (hi_large - lo_large)


class TestEffectSize:
    def test_cohens_d_known_value(self):
        d = cohens_d(0.0, 1.0, 1.0, 1.0, 100, 100)
        assert d == pytest.approx(1.0)

    def test_cohens_d_zero_variance(self):
        assert cohens_d(1.0, 1.0, 0.0, 0.0, 10, 10) == 0.0


class TestRelativeLift:
    def test_percent(self):
        assert relative_lift(0.05, 1.0) == pytest.approx(5.0)

    def test_zero_control_mean(self):
        assert relative_lift(0.05, 0.0) == 0.0


class TestLeakageCheck:
    def test_warns_on_imbalance(self):
        rng = np.random.default_rng(2)
        X = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(2.0, 1.0, 200)])
        treatment = np.concatenate([np.zeros(200, int), np.ones(200, int)])
        with pytest.warns(UserWarning, match="leakage"):
            assert check_leakage(X, treatment) is True

    def test_silent_when_balanced(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0.0, 1.0, 400)
        treatment = np.concatenate([np.zeros(200, int), np.ones(200, int)])
        assert check_leakage(X, treatment) is False
