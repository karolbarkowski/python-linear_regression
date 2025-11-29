from enum import Enum
from models.data_params import DataGenerationParams
from regression_steps import perform_regression
from steps.visualize import visualize

class Mode(Enum):
    SKLEARN = "sklearn"
    MANUAL = "manual"

process_config = DataGenerationParams(
    n_samples=100,
    slope=2,
    intercept=1,
    noise_std=1,
    random_seed=42,
    training_fraction=0.8
)

manual_output = perform_regression(process_config, mode=Mode.MANUAL.value)
sklearn_output = perform_regression(process_config, mode=Mode.SKLEARN.value)

# VISUALIZATIONS
visualize(manual_output, sklearn_output, process_config)
