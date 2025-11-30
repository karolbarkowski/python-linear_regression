"""
Manual linear regression training using gradient descent
"""
import numpy as np
from models.input_data import InputData
from models.training_params import TrainingParams
from models.training_result import TrainingResult


def train_manual(input_data: InputData, params: TrainingParams) -> TrainingResult:
    """
    Train a linear regression model using gradient descent.

    Implements the gradient descent algorithm to learn the optimal slope and
    intercept for a linear relationship between X and y.

    Args:
        input_data: Training and test data containing X_train, y_train, X_test, y_test
        params: Training configuration (iterations, learning rate, loss threshold)

    Returns:
        TrainingResult containing learned slope, intercept, and predictions on test data
    """
    # Get the number of training samples
    n_samples = input_data.X_train.shape[0]

    # Initialize parameters with small random values
    slope = np.random.randn() * 0.01  # random slope close to 0
    intercept = np.random.randn() * 0.01  # random intercept close to 0

    prev_loss = None

    # Gradient Descent Loop
    for iteration in range(params.n_iterations):
        # STEP 1: Forward Pass - Make predictions
        # Formula: y_pred = slope * X + intercept
        y_pred = slope * input_data.X_train.flatten() + intercept

        # STEP 2: Calculate Loss (Mean Squared Error)
        # MSE = (1/n) * sum((y_true - y_pred)^2)
        # This measures how far off our predictions are
        loss = np.mean((input_data.y_train - y_pred) ** 2)

        # Check if loss change is below threshold and stop if so (early stopping)
        if prev_loss is not None and abs(prev_loss - loss) < params.loss_threshold:
            break

        prev_loss = loss

        # STEP 3: Calculate Gradients
        # Gradient tells us which direction to adjust parameters to reduce the loss

        # Gradient of loss with respect to slope:
        # dL/d(slope) = -(2/n) * sum(X * (y_true - y_pred))
        gradient_slope = -(2 / n_samples) * np.sum(
            input_data.X_train.flatten() * (input_data.y_train - y_pred)
        )

        # Gradient of loss with respect to intercept:
        # dL/d(intercept) = -(2/n) * sum(y_true - y_pred)
        gradient_intercept = -(2 / n_samples) * np.sum(input_data.y_train - y_pred)

        # STEP 4: Update Parameters
        # Move in the opposite direction of the gradient
        # (gradient points uphill, we want to go downhill)
        slope = slope - params.learning_rate * gradient_slope
        intercept = intercept - params.learning_rate * gradient_intercept

    # Make predictions on test data
    predictions = slope * input_data.X_test.flatten() + intercept

    return TrainingResult(slope=slope, intercept=intercept, predictions=predictions)
