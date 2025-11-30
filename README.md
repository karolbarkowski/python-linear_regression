# Linear Regression

A Python implementation comparing manual gradient descent with scikit-learn's linear regression.

## Overview

This project demonstrates linear regression using two approaches:
- **Manual implementation** using gradient descent
- **Scikit-learn** implementation using the closed-form solution

The project generates synthetic linear data with noise, trains both models, and visualizes the results side-by-side.


## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- scikit-learn

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy matplotlib scikit-learn
```

## Usage

Run the main script to train both models and visualize the results:

```bash
python main.py
```

The script will:
1. Generate synthetic linear data with configurable parameters
2. Train a manual gradient descent model
3. Train a scikit-learn linear regression model
4. Print the learned equations and training times
5. Display a 2x2 comparison plot showing:
   - Learned lines vs true relationship
   - Prediction errors on test data

## Configuration

You can modify the data generation and training parameters in [main.py](main.py):

```python
# Data generation
data_config = DataGenerationParams(
    n_samples=100,           # Number of total samples
    slope=2,                 # True slope
    intercept=1,             # True intercept
    noise_std=1,             # Standard deviation of noise
    random_seed=42,          # For reproducibility
    training_fraction=0.8    # 80% train, 20% test
)

# Manual gradient descent
training_config = TrainingParams(
    n_iterations=1000,       # Maximum iterations
    learning_rate=0.01,      # Step size
    loss_threshold=1e-6      # Early stopping threshold
)
```

## How It Works

### Manual Gradient Descent

The manual implementation in [steps/train_manual.py](steps/train_manual.py) uses the gradient descent algorithm:

1. Initialize slope and intercept with random values
2. For each iteration:
   - Make predictions: `y_pred = slope * X + intercept`
   - Calculate loss (MSE): `loss = mean((y_true - y_pred)²)`
   - Compute gradients of loss with respect to parameters
   - Update parameters: `param = param - learning_rate * gradient`
3. Stop when loss change is below threshold or max iterations reached

### Scikit-learn Implementation

The sklearn implementation in [steps/train_sklearn.py](steps/train_sklearn.py) uses the normal equation (closed-form solution) to find the optimal parameters directly without iteration.

