"""
Visualization of regression results
"""
from matplotlib import pyplot as plt
import numpy as np
from models import DataGenerationParams, RegressionOutput, InputData


def visualize(
    manual_output: RegressionOutput,
    sklearn_output: RegressionOutput,
    input_data: InputData,
    data_config: DataGenerationParams
) -> None:
    """
    Visualize and compare manual and sklearn regression results.

    Creates a 2x2 grid of plots comparing both training methods:
    - Left column: Learned lines vs true relationship
    - Right column: Prediction errors on test data

    Args:
        manual_output: Results from manual gradient descent training
        sklearn_output: Results from sklearn training
        input_data: Original training and test data
        data_config: Original data generation parameters (for true line plotting)
    """
    # Create figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig.suptitle('Linear Regression Visualizations: Manual vs Sklearn', fontsize=16, fontweight='bold')

    # Process both outputs
    outputs = [
        (manual_output, 'Manual', 0),
        (sklearn_output, 'Sklearn', 1)
    ]

    for output, method_name, row in outputs:
        # Extract data from output
        X = np.concatenate([input_data.X_train, input_data.X_test])
        X_train = input_data.X_train
        y_train = input_data.y_train
        X_test = input_data.X_test
        y_test = input_data.y_test
        y_test_pred = output.predictions
        learned_slope = output.learned_slope
        learned_intercept = output.learned_intercept

        # Plot 1: All Data with Learned Line (left column)
        ax1 = axes[row, 0]

        # Scatter plot of all data points
        ax1.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data', s=40)
        ax1.scatter(X_test, y_test, color='green', alpha=0.6, label='Test data', s=40)

        # Plot the learned line
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred_line = learned_slope * X_line.flatten() + learned_intercept
        ax1.plot(X_line, y_pred_line, color='red', linewidth=2.5, label='Learned line', zorder=5)

        # Plot the true line for comparison
        y_true_line = data_config.slope * X_line.flatten() + data_config.intercept
        ax1.plot(X_line, y_true_line, color='orange', linewidth=2, linestyle='--',
                 label='True line', alpha=0.8, zorder=4)

        ax1.set_xlabel('X (Independent Variable)', fontsize=11)
        ax1.set_ylabel('y (Dependent Variable)', fontsize=11)
        ax1.set_title(f'{method_name}: Learned vs True Relationship', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Prediction Errors on Test Data (right column)
        ax2 = axes[row, 1]

        # Scatter plot of test data
        ax2.scatter(X_test, y_test, color='green', alpha=0.6, label='Actual test values', s=60, zorder=3)
        ax2.scatter(X_test, y_test_pred, color='red', marker='x', s=80,
                    label='Predicted test values', linewidths=2, zorder=4)

        # Draw lines showing the distance between actual and predicted
        for i in range(len(X_test)):
            ax2.plot([X_test[i], X_test[i]], [y_test[i], y_test_pred[i]],
                    color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=2)

        ax2.set_xlabel('X (Independent Variable)', fontsize=11)
        ax2.set_ylabel('y (Dependent Variable)', fontsize=11)
        ax2.set_title(f'{method_name}: Actual vs Predicted (with error distances)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
