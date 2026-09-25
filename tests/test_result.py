"""Unit tests for CUPEDResult."""

import pytest

from cuped_lite.result import CUPEDResult


def make_result(**overrides):
    defaults = {
        "theta": 0.8,
        "theta_strategy": "pooled",
        "correlation": 0.85,
        "variance_reduction": 0.2775,
        "control_mean_original": 10.0,
        "treatment_mean_original": 10.15,
        "control_mean_adjusted": 10.0,
        "treatment_mean_adjusted": 10.15,
        "control_variance_original": 4.0,
        "treatment_variance_original": 4.0,
        "control_variance_adjusted": 1.11,
        "treatment_variance_adjusted": 1.11,
        "original_difference": 0.15,
        "adjusted_difference": 0.15,
        "effect_size": 0.075,
        "relative_lift": 1.5,
        "p_value": 6.7e-6,
        "t_statistic": 4.5,
        "degrees_of_freedom": 4998.0,
        "standard_error": 0.0333,
        "ci_lower": 0.0847,
        "ci_upper": 0.2153,
        "alpha": 0.05,
        "n_control": 2500,
        "n_treatment": 2500,
        "theta_correction": True,
        "warnings": [],
    }
    defaults.update(overrides)
    return CUPEDResult(**defaults)


class TestProperties:
    def test_n_total(self):
        assert make_result().n_total == 5000

    def test_absolute_effect(self):
        assert make_result().absolute_effect == 0.15

    def test_variance_reduction_percent(self):
        assert make_result().variance_reduction_percent == pytest.approx(27.75)

    def test_significant(self):
        assert make_result().significant is True
        assert make_result(p_value=0.5).significant is False


class TestSerialization:
    def test_to_dict(self):
        data = make_result().to_dict()
        assert data["n_total"] == 5000
        assert data["absolute_effect"] == 0.15
        assert data["variance_reduction_percent"] == pytest.approx(27.75)
        assert data["significant"] is True
        assert "p_value" in data

    def test_to_dict_handles_per_group_theta(self):
        data = make_result(theta={0: 0.8, 1: 0.7}, theta_strategy="per_group").to_dict()
        assert data["theta"] == {0: 0.8, 1: 0.7}


class TestFormatting:
    def test_to_text(self):
        text = make_result().to_text()
        assert "CUPED Analysis Summary" in text
        assert "p-value" in text
        assert "Variance reduction" in text
        assert "Relative lift" in text

    def test_repr(self):
        assert "CUPEDResult" in repr(make_result())
