from enum import Enum
from models import DataGenerationParams
from regression_steps import perform_regression
from steps import generate_data, visualize

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

# Generate some random training/test data
input_data = generate_data(process_config)

# TRAINING
manual_output = perform_regression(input_data, mode=Mode.MANUAL.value)
sklearn_output = perform_regression(input_data, mode=Mode.SKLEARN.value)

print("MANUAL REGRESSION RESULTS")
print(f"Learned equation: y = {manual_output.learned_slope:.2f}x + {manual_output.learned_intercept:.2f}")
print(f"Performed in: {manual_output.ticks_taken}.")
print()

print("SKLEARN REGRESSION RESULTS")
print(f"Learned equation: y = {sklearn_output.learned_slope:.2f}x + {sklearn_output.learned_intercept:.2f}")
print(f"Performed in: {sklearn_output.ticks_taken}.")
print()

# VISUALIZATION
visualize(manual_output, sklearn_output, input_data, process_config)
