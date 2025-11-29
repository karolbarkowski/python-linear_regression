from sklearn.linear_model import LinearRegression

from models.input_data import InputData


def train_sklearn(input_data: InputData):
    model = LinearRegression()
    model.fit(input_data.X_train, input_data.y_train)

    learned_slope = model.coef_[0]  
    learned_intercept = model.intercept_  

    return learned_slope, learned_intercept, model.predict(input_data.X_test)