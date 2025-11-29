import numpy as np


class ManualLinearRegressionModel:
    """
    Simple wrapper class that mimics scikit-learn's LinearRegression interface.

    This allows the manual implementation to be used interchangeably with
    sklearn's model in the rest of the code
    """
    def __init__(self, slope, intercept):
        self.coef_ = np.array([slope])  # Stored as array to match sklearn format
        self.intercept_ = intercept  # Match sklearn's attribute name
        self.slope = slope  # Keep as simple float for convenience

    def predict(self, X):
        return self.slope * X.flatten() + self.intercept_


def train_manual(X_train, y_train, n_iterations=1000, learning_rate=0.01):
    loss_history = []

    # Get the number of training samples
    n_samples = X_train.shape[0]

    # Initialize parameters with small random values
    learned_slope = np.random.randn() * 0.01  # random slope close to 0
    learned_intercept = np.random.randn() * 0.01    # random intercept close to 0

    # Initialize loss to 0 (will be updated in the loop)
    loss = 0.0

    # Gradient Descent Loop
    for iteration in range(n_iterations):
        # STEP 1: Forward Pass - Make predictions
        # Formula: y_pred = weight * X + bias
        y_pred = learned_slope * X_train.flatten() + learned_intercept

        # STEP 2: Calculate Loss (Mean Squared Error)
        # MSE = (1/n) * sum((y_true - y_pred)^2)
        # This measures how far off our predictions are
        loss = np.mean((y_train - y_pred) ** 2)
        loss_history.append(loss)

        # STEP 3: Calculate derivatives
        # Gradient tells us which direction to adjust parameters
        # to reduce the loss

        # Gradient of loss with respect to weight:
        # dL/dw = -(2/n) * sum(X * (y_true - y_pred))
        gradient_weight = -(2 / n_samples) * np.sum(X_train.flatten() * (y_train - y_pred))

        # Gradient of loss with respect to bias:
        # dL/db = -(2/n) * sum(y_true - y_pred)
        gradient_bias = -(2 / n_samples) * np.sum(y_train - y_pred)

        # STEP 4: Update Parameters
        # Move in the opposite direction of the gradient
        # (gradient points uphill, we want to go downhill)
        learned_slope = learned_slope - learning_rate * gradient_weight
        learned_intercept = learned_intercept - learning_rate * gradient_bias

        
    # Create a model object that has a predict() method
    # This makes it compatible with sklearn models
    model = ManualLinearRegressionModel(learned_slope, learned_intercept)

    return model, learned_slope, learned_intercept
