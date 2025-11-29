import numpy as np
from models.input_data import InputData

def train_manual(input_data: InputData, n_iterations=1000, learning_rate=0.01, loss_threshold=1e-6):
    loss_history = []

    # Get the number of training samples
    n_samples = input_data.X_train.shape[0]

    # Initialize parameters with small random values
    learned_slope = np.random.randn() * 0.01  # random slope close to 0
    learned_intercept = np.random.randn() * 0.01    # random intercept close to 0

    # Initialize loss to 0 (will be updated in the loop)
    loss = 0.0
    prev_loss = None

    # Gradient Descent Loop
    for iteration in range(n_iterations):
        # STEP 1: Forward Pass - Make predictions
        # Formula: y_pred = weight * X + bias
        y_pred = learned_slope * input_data.X_train.flatten() + learned_intercept

        # STEP 2: Calculate Loss (Mean Squared Error)
        # MSE = (1/n) * sum((y_true - y_pred)^2)
        # This measures how far off our predictions are
        loss = np.mean((input_data.y_train - y_pred) ** 2)
        loss_history.append(loss)

        # check if loss change is below threshold and stop if so
        if prev_loss is not None and abs(prev_loss - loss) < loss_threshold:
            break

        prev_loss = loss

        # STEP 3: Calculate derivatives
        # Gradient tells us which direction to adjust parameters
        # to reduce the loss

        # Gradient of loss with respect to weight:
        # dL/dw = -(2/n) * sum(X * (y_true - y_pred))
        gradient_weight = -(2 / n_samples) * np.sum(input_data.X_train.flatten() * (input_data.y_train - y_pred))

        # Gradient of loss with respect to bias:
        # dL/db = -(2/n) * sum(y_true - y_pred)
        gradient_bias = -(2 / n_samples) * np.sum(input_data.y_train - y_pred)

        # STEP 4: Update Parameters
        # Move in the opposite direction of the gradient
        # (gradient points uphill, we want to go downhill)
        learned_slope = learned_slope - learning_rate * gradient_weight
        learned_intercept = learned_intercept - learning_rate * gradient_bias

    return learned_slope, learned_intercept, learned_slope * input_data.X_test.flatten() + learned_intercept
