from sklearn.linear_model import LinearRegression


def train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    learned_slope = model.coef_[0]  
    learned_intercept = model.intercept_  
    return model, learned_slope, learned_intercept