from models import DataGenerationParams
from models.regression_output import RegressionOutput
from steps import generate_linear_data, split_data, train_manual, train_sklearn, visualize


def perform_regression(process_config: DataGenerationParams, mode: str = "manual"):
    # GENERATING SAMPLE DATA
    X, Y = generate_linear_data(process_config)

    print(f"Generated {len(X)} data points for equation: y = {process_config.slope}x + {process_config.intercept}")
    print()

    # SPLITTING DATA
    X_train, y_train, X_test, y_test = split_data(X, Y, process_config.training_fraction)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    print()

    # TRAINING THE MODEL
    match mode:
        case "manual":
            model, learned_slope, learned_intercept = train_manual(X_train, y_train, n_iterations=1000, learning_rate=0.01)
        case "sklearn":
            model, learned_slope, learned_intercept = train_sklearn(X_train, y_train)
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    print(f"Learned equation: y = {learned_slope:.2f}x + {learned_intercept:.2f}")
    print()

    # PREDICTIONS
    y_test_pred = model.predict(X_test)

    output = RegressionOutput(
        mode=mode,
        X=X,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_test_pred=y_test_pred,
        learned_slope=float(learned_slope),
        learned_intercept=float(learned_intercept)
    )

    return output  
    


  