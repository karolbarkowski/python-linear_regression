from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class RegressionOutput:
    mode: str
    predictions: NDArray[np.float64]
    learned_slope: float
    learned_intercept: float
    ticks_taken: float = 0.0
