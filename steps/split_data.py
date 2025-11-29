def split_data(X, Y, training_fraction):
   
    split_index = int(training_fraction * len(X))

    X_train = X[:split_index] 
    y_train = Y[:split_index]
    X_test = X[split_index:] 
    y_test = Y[split_index:]

    return X_train, y_train, X_test, y_test