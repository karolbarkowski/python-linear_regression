from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def visualize(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, model, process_config, X):
    # figure with custom grid layout - GridSpec allows to create flexible subplot arrangements
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Linear Regression Visualizations', fontsize=16, fontweight='bold')

    # Plot 1: Training Data and Fitted Line (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data', s=30)
    ax1.plot(X_train, y_train_pred, color='red', linewidth=2, label='Fitted line')
    ax1.set_xlabel('X (Independent Variable)', fontsize=10)
    ax1.set_ylabel('y (Dependent Variable)', fontsize=10)
    ax1.set_title('Training Data with Fitted Line', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: All Data Together (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(X_train, y_train, color='blue', alpha=0.4, label='Training data', s=20)
    ax2.scatter(X_test, y_test, color='green', alpha=0.4, label='Test data', s=20)
    ax2.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted line')
    # Also plot the true line for comparison
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true_line = process_config.slope * X_line + process_config.intercept
    ax2.plot(X_line, y_true_line, color='orange', linewidth=2, linestyle='--', label='True line')
    ax2.set_xlabel('X (Independent Variable)', fontsize=10)
    ax2.set_ylabel('y (Dependent Variable)', fontsize=10)
    ax2.set_title('Complete Dataset: Learned vs True Relationship', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residual Plot
    # gs[1, :] means row 1, all columns
    ax3 = fig.add_subplot(gs[1, :])
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    ax3.scatter(y_train_pred, residuals_train, color='blue', alpha=0.4, label='Training residuals', s=20)
    ax3.scatter(y_test_pred, residuals_test, color='green', alpha=0.4, label='Test residuals', s=20)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax3.set_xlabel('Predicted Values', fontsize=10)
    ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=10)
    ax3.set_title('Residual Plot (should be random around 0)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()