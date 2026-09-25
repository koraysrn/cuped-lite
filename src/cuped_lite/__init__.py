"""cuped-lite: CUPED variance reduction for A/B tests.

``cuped-lite`` reduces the variance of experiment metrics by exploiting
pre-experiment data, so that the same statistical power can be reached with a
smaller sample. It depends only on ``numpy`` and ``scipy``.
"""

from __future__ import annotations

from ._power import SampleSizeResult, required_sample_size
from .cuped import CUPED
from .result import CUPEDResult

__all__ = ["CUPED", "CUPEDResult", "SampleSizeResult", "required_sample_size"]

__version__ = "0.1.0"
