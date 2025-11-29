import numpy as np
import matplotlib.pyplot as plt
from models.data_params import DataGenerationParams

def generate_linear_data(params: DataGenerationParams):
    np.random.seed(params.random_seed)

    # Generate x values
    X = np.linspace(0, 10, params.n_samples)

    # Generate y values based on the linear equation with added noise
    noise = np.random.normal(0, params.noise_std, params.n_samples)  # Gaussian noise
    y = params.slope * X + params.intercept + noise

    # Reshape X from 1D array (100,) to 2D array (100, 1) as scikit-learn requires 2D arrays 
    X = X.reshape(-1, 1)

    return X, y
