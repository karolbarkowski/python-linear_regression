"""
Sklearn-based linear regression training
"""
from sklearn.linear_model import LinearRegression
from models.input_data import InputData
from models.training_result import TrainingResult


def train_sklearn(input_data: InputData) -> TrainingResult:
    """
    Train a linear regression model using scikit-learn's LinearRegression.

    Uses sklearn's optimized implementation which finds the exact solution
    using the normal equation (closed-form solution).

    Args:
        input_data: Training and test data containing X_train, y_train, X_test, y_test

    Returns:
        TrainingResult containing learned slope, intercept, and predictions on test data
    """
    model = LinearRegression()
    model.fit(input_data.X_train, input_data.y_train)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    predictions = model.predict(input_data.X_test)

    return TrainingResult(slope=slope, intercept=intercept, predictions=predictions)