"""
Training result structure for regression models
"""
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray


class TrainingResult(NamedTuple):
    """Result from training a linear regression model"""
    slope: float
    intercept: float
    predictions: NDArray[np.float64]
