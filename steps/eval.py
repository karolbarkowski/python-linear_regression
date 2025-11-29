from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    # Mean Squared Error (the lower, the better)
    train_mse = mean_squared_error(y_train, y_train_pred) 
    test_mse = mean_squared_error(y_test, y_test_pred)
    # R-squared (R²) Score (1 = perfect fit, 0 = model is no better than predicting the mean)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2