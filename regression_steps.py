"""
Regression orchestration - coordinates training and performance measurement
"""
import time
from models import Mode, TrainingParams
from models.input_data import InputData
from models.regression_output import RegressionOutput
from steps import train_manual, train_sklearn


def perform_regression(
    input_data: InputData,
    mode: Mode,
    training_params: TrainingParams | None = None
) -> RegressionOutput:
    """
    Perform linear regression training and measure performance.

    Args:
        input_data: Training and test data
        mode: Training mode (MANUAL or SKLEARN)
        training_params: Parameters for manual training (required if mode is MANUAL)

    Returns:
        RegressionOutput containing predictions, learned parameters, and timing

    Raises:
        ValueError: If mode is MANUAL but training_params is None
    """
    # Start performance measurement
    start_time = time.perf_counter()

    # Model training based on mode
    match mode:
        case Mode.MANUAL:
            if training_params is None:
                raise ValueError("training_params required for MANUAL mode")
            result = train_manual(input_data, training_params)
        case Mode.SKLEARN:
            result = train_sklearn(input_data)
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    # End performance measurement
    end_time = time.perf_counter()

    output = RegressionOutput(
        mode=mode,
        predictions=result.predictions,
        learned_slope=result.slope,
        learned_intercept=result.intercept,
        ticks_taken=end_time - start_time
    )

    return output
