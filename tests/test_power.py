"""Unit tests for the power/sample-size module."""

import numpy as np
import pytest
from scipy import stats

from cuped_lite._power import SampleSizeResult, required_sample_size


def test_known_formula():
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    expected = 2 * z**2
    assert required_sample_size(1.0, 1.0) == int(np.ceil(expected))


def test_variance_reduction_reduces_sample_size():
    n0 = required_sample_size(0.2, 1.0, variance_reduction=0.0)
    n1 = required_sample_size(0.2, 1.0, variance_reduction=0.5)
    assert n1 < n0


def test_larger_power_requires_more_samples():
    n80 = required_sample_size(0.2, 1.0, power=0.8)
    n90 = required_sample_size(0.2, 1.0, power=0.9)
    assert n90 > n80


def test_ratio_scaling():
    n1 = required_sample_size(0.2, 1.0, ratio=1.0)
    n2 = required_sample_size(0.2, 1.0, ratio=2.0)
    # A 1:2 allocation requires fewer control observations than 1:1.
    assert n2 < n1


def test_negative_diff_uses_absolute_value():
    assert required_sample_size(-0.2, 1.0) == required_sample_size(0.2, 1.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        required_sample_size(0.0, 1.0)
    with pytest.raises(ValueError):
        required_sample_size(0.2, 0.0)
    with pytest.raises(ValueError):
        required_sample_size(0.2, 1.0, variance_reduction=1.5)
    with pytest.raises(ValueError):
        required_sample_size(0.2, 1.0, alpha=1.5)
    with pytest.raises(ValueError):
        required_sample_size(0.2, 1.0, power=1.5)


def test_sample_size_result_to_dict():
    result = SampleSizeResult(100, 80, 20, 20.0, 0.8, 0.05, 0.2, 1.0, 0.5)
    data = result.to_dict()
    assert data["n_saved"] == 20
    assert data["savings_percent"] == 20.0
