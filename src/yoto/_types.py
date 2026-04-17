"""NumPy array type aliases used across the YOTO package."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

GrayImage: TypeAlias = npt.NDArray[np.uint8]
Image: TypeAlias = npt.NDArray[np.uint8]
BBoxArray: TypeAlias = npt.NDArray[np.float64]
FloatArray: TypeAlias = npt.NDArray[np.float32]
