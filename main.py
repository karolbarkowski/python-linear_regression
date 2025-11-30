from models import DataGenerationParams, Mode, TrainingParams
from regression_steps import perform_regression
from steps import generate_data, visualize

# Data generation configuration
data_config = DataGenerationParams(
    n_samples=100,
    slope=2,
    intercept=1,
    noise_std=1,
    random_seed=42,
    training_fraction=0.8
)

# Training configuration for manual gradient descent
training_config = TrainingParams(
    n_iterations=1000,
    learning_rate=0.01,
    loss_threshold=1e-6
)

# Generate some random training/test data
input_data = generate_data(data_config)

# TRAINING
manual_output = perform_regression(input_data, mode=Mode.MANUAL, training_params=training_config)
sklearn_output = perform_regression(input_data, mode=Mode.SKLEARN)

print("MANUAL REGRESSION RESULTS")
print(f"Learned equation: y = {manual_output.learned_slope:.2f}x + {manual_output.learned_intercept:.2f}")
print(f"Performed in: {manual_output.ticks_taken}.")
print()

print("SKLEARN REGRESSION RESULTS")
print(f"Learned equation: y = {sklearn_output.learned_slope:.2f}x + {sklearn_output.learned_intercept:.2f}")
print(f"Performed in: {sklearn_output.ticks_taken}.")
print()

# VISUALIZATION
visualize(manual_output, sklearn_output, input_data, data_config)
