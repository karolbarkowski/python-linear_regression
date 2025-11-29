import time
from models.input_data import InputData
from models.regression_output import RegressionOutput
from steps import train_manual, train_sklearn


def perform_regression(input_data: InputData, mode: str = "manual"):
    # Start performance measurement
    start_time = time.perf_counter()

    # Model training based on mode
    match mode:
        case "manual":
            learned_slope, learned_intercept, predictions = train_manual(input_data, n_iterations=1000, learning_rate=0.01)
        case "sklearn":
            learned_slope, learned_intercept, predictions = train_sklearn(input_data)
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    # End performance measurement
    end_time = time.perf_counter()

    output = RegressionOutput(
        mode=mode,
        predictions=predictions,
        learned_slope=float(learned_slope),
        learned_intercept=float(learned_intercept),
        ticks_taken=end_time - start_time
    )

    return output  
    


  