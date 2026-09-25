"""Unit tests for input validation."""

import numpy as np
import pytest

from cuped_lite.validation import (
    validate_fit_inputs,
    validate_transform_inputs,
    validate_treatment,
)


class TestAsNumericVector:
    def test_accepts_int_and_float(self):
        X = np.array([1, 2, 3, 4])
        Y = np.array([1.5, 2.5, 3.5, 4.5])
        t = np.array([0, 0, 1, 1])
        Xa, Ya, ta = validate_fit_inputs(X, Y, t)
        assert Xa.dtype == np.float64
        assert Ya.dtype == np.float64
        assert ta.dtype == np.int64

    def test_rejects_bool_x(self):
        with pytest.raises(TypeError, match="numeric"):
            validate_fit_inputs(
                np.array([True, False, True, False]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_object_x(self):
        with pytest.raises(TypeError, match="numeric"):
            validate_fit_inputs(
                np.array(["a", "b", "c", "d"], dtype=object),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_fit_inputs(
                np.array([1.0, np.nan, 3.0, 4.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="infinite"):
            validate_fit_inputs(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, np.inf, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            validate_fit_inputs(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
            )


class TestValidateTreatment:
    def test_bool_mapped_to_binary(self):
        t = validate_treatment(np.array([False, False, True, True]))
        np.testing.assert_array_equal(t, np.array([0, 0, 1, 1]))

    def test_rejects_single_class(self):
        with pytest.raises(ValueError, match="exactly two classes"):
            validate_treatment(np.array([0, 0, 0, 0]))

    def test_rejects_three_classes(self):
        with pytest.raises(ValueError, match="two classes"):
            validate_treatment(np.array([0, 1, 2, 0]))


class TestFitInputs:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            validate_fit_inputs(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
            )

    def test_too_few_total_observations(self):
        with pytest.raises(ValueError, match="At least 4"):
            validate_fit_inputs(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([0, 1, 1]),
            )

    def test_too_few_per_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_fit_inputs(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 1, 1, 1]),
            )


class TestTransformInputs:
    def test_ok_without_treatment(self):
        X = np.array([1.0, 2.0, 3.0])
        Y = np.array([2.0, 3.0, 4.0])
        Xa, _, ta = validate_transform_inputs(X, Y)
        assert Xa.shape == (3,)
        assert ta is None

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            validate_transform_inputs(np.array([1.0, 2.0]), np.array([1.0]))
