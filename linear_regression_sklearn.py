from models import DataGenerationParams
from steps import generate_linear_data, split_data, train, evaluate_model, visualize

process_config = DataGenerationParams(
    n_samples=100,
    slope=2,
    intercept=1,
    noise_std=1,
    random_seed=42,
    training_fraction=0.8
)

# GENERATING SAMPLE DATA
X, Y = generate_linear_data(process_config)

print(f"Generated {len(X)} data points for equation: y = {process_config.slope}x + {process_config.intercept}")
print()

# SPLITTING DATA
X_train, y_train, X_test, y_test = split_data(X, Y, process_config.training_fraction)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print()

# TRAINING THE MODEL
model, learned_slope, learned_intercept = train(X_train, y_train)

print(f"Learned equation: y = {learned_slope:.2f}x + {learned_intercept:.2f}")
print()

# PREDICTIONS
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# MODEL EVALUATION
train_mse, test_mse, train_r2, test_r2 = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)

# VISUALIZATIONS
visualize(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, model, process_config, X)