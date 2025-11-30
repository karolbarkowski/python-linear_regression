"""
Linear data generation for regression training
"""
import numpy as np
from models.data_params import DataGenerationParams
from models.input_data import InputData


def generate_data(params: DataGenerationParams) -> InputData:
    """
    Generate synthetic linear data with Gaussian noise.

    Creates a dataset following y = slope * x + intercept + noise,
    then splits it into training and test sets.

    Args:
        params: Configuration for data generation including slope, intercept,
                noise level, sample count, and train/test split ratio

    Returns:
        InputData containing X_train, y_train, X_test, y_test arrays
    """
    np.random.seed(params.random_seed)

    # Generate x values
    X = np.linspace(0, 10, params.n_samples)

    # Generate y values based on the linear equation with added noise
    noise = np.random.normal(0, params.noise_std, params.n_samples)  # Gaussian noise
    y = params.slope * X + params.intercept + noise

    # Reshape X from 1D array (100,) to 2D array (100, 1) as scikit-learn requires 2D arrays 
    X = X.reshape(-1, 1)

    split_index = int(params.training_fraction * len(X))

    return InputData(
        X_train=X[:split_index],
        y_train=y[:split_index],
        X_test=X[split_index:],
        y_test=y[split_index:]
    )
