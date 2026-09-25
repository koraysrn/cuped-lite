"""Input validation helpers for cuped-lite.

All validation failures are raised as ``ValueError`` or ``TypeError`` (hard
errors). Conditions that are statistically suspicious but not strictly invalid
are reported via ``warnings.warn`` (soft warnings) by the caller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EPSILON",
    "MIN_GROUP_SIZE",
    "validate_fit_inputs",
    "validate_transform_inputs",
    "validate_treatment",
]

#: Minimum number of observations required in each group (Welch t-test precondition).
MIN_GROUP_SIZE = 2

#: Threshold below which a variance is treated as numerically zero.
EPSILON = 1e-12

#: NumPy dtype kinds accepted as numeric: bool, signed int, unsigned int, float.
_FLOAT_KINDS = "biuf"


def _as_numeric_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Coerce ``values`` into a finite 1-D float64 NumPy array.

    Raises ``TypeError`` for non-numeric input and ``ValueError`` for
    non-finite or non-vector input.
    """
    arr = np.asarray(values)
    if arr.dtype == np.dtype("bool"):
        raise TypeError(f"`{name}` must be numeric (int/float); got bool.")
    if arr.dtype.kind not in _FLOAT_KINDS:
        raise TypeError(f"`{name}` must be numeric (int/float); got dtype {arr.dtype}.")
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be a 1-D vector; got {arr.ndim} dimension(s).")
    out = arr.astype(np.float64, copy=False)
    if np.any(~np.isfinite(out)):
        raise ValueError(f"`{name}` must not contain NaN or infinite values.")
    return out


def validate_treatment(treatment: ArrayLike) -> NDArray[np.int64]:
    """Validate a binary treatment/group flag vector.

    Returns an int64 array that contains exactly two distinct classes,
    ``{0, 1}``. Boolean input is accepted and mapped to 0/1.
    """
    arr = np.asarray(treatment)
    if arr.dtype == np.dtype("bool"):
        arr = arr.astype(np.int64)
    if arr.dtype.kind not in _FLOAT_KINDS:
        raise TypeError(f"`treatment` must be numeric (int/float); got dtype {arr.dtype}.")
    if arr.ndim != 1:
        raise ValueError(f"`treatment` must be a 1-D vector; got {arr.ndim} dimension(s).")

    arr_f = arr.astype(np.float64, copy=False)
    if np.any(~np.isfinite(arr_f)):
        raise ValueError("`treatment` must not contain NaN or infinite values.")

    unique = np.unique(np.round(arr_f, 6))
    if not set(unique.tolist()).issubset({0.0, 1.0}):
        raise ValueError(
            "`treatment` must contain only two classes encoded as 0 (control) "
            f"and 1 (variant); got classes {unique.tolist()}."
        )
    if unique.size != 2:
        raise ValueError(
            "`treatment` must contain exactly two classes (0 and 1); "
            f"got {unique.size} distinct value(s)."
        )
    return arr.astype(np.int64)


def validate_fit_inputs(
    X: ArrayLike, Y: ArrayLike, treatment: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Validate and coerce the three inputs of :meth:`CUPED.fit`.

    Returns ``(X, Y, treatment)`` as float64/float64/int64 vectors.
    """
    X_arr = _as_numeric_vector(X, "X")
    Y_arr = _as_numeric_vector(Y, "Y")
    treatment_arr = validate_treatment(treatment)

    n = X_arr.shape[0]
    if Y_arr.shape[0] != n or treatment_arr.shape[0] != n:
        raise ValueError(
            "`X`, `Y` and `treatment` must have the same length; got "
            f"{n}, {Y_arr.shape[0]} and {treatment_arr.shape[0]}."
        )
    if n < MIN_GROUP_SIZE * 2:
        raise ValueError(
            f"At least {MIN_GROUP_SIZE * 2} observations are required in total; got {n}."
        )

    n_control = int(np.sum(treatment_arr == 0))
    n_variant = int(np.sum(treatment_arr == 1))
    if n_control < MIN_GROUP_SIZE or n_variant < MIN_GROUP_SIZE:
        raise ValueError(
            f"Each group must contain at least {MIN_GROUP_SIZE} observations "
            f"(Welch t-test precondition); got control={n_control}, variant={n_variant}."
        )
    return X_arr, Y_arr, treatment_arr


def validate_transform_inputs(
    X: ArrayLike, Y: ArrayLike, treatment: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64] | None]:
    """Validate inputs for :meth:`CUPED.transform`.

    ``treatment`` is optional and is only required when
    ``strategy="per_group"``.
    """
    X_arr = _as_numeric_vector(X, "X")
    Y_arr = _as_numeric_vector(Y, "Y")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError(
            f"`X` and `Y` must have the same length; got {X_arr.shape[0]} and {Y_arr.shape[0]}."
        )

    treatment_arr: NDArray[np.int64] | None = None
    if treatment is not None:
        treatment_arr = validate_treatment(treatment)
        if treatment_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                "`treatment` must have the same length as `X` and `Y`; got "
                f"{treatment_arr.shape[0]} and {X_arr.shape[0]}."
            )
    return X_arr, Y_arr, treatment_arr
