import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    train_mae = mean_absolute_error(
        y_train,
        y_train_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        y_test_pred
    )

    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

    return metrics


def plot_predictions(y_test, y_pred, fd_name):

    plt.figure(figsize=(10, 4))

    plt.plot(
        y_test,
        label='Actual RUL'
    )

    plt.plot(
        y_pred,
        label='Predicted RUL'
    )

    plt.title(f'{fd_name} Predictions')

    plt.legend()

    plt.show()


def get_feature_importance(
    model,
    feature_names,
    top_k=10
):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    return (
        importance
        .sort_values(ascending=False)
        .head(top_k)
    )