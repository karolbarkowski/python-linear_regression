from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class RegressionOutput:
    mode: str
    X: NDArray[np.float64]
    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.float64]
    y_test_pred: NDArray[np.float64]
    learned_slope: float
    learned_intercept: float
