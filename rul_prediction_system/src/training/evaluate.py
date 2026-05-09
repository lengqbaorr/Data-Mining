import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)


def _add_regression_metrics(metrics, prefix, y_true, y_pred):

    metrics[f'{prefix}_rmse'] = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    metrics[f'{prefix}_mae'] = mean_absolute_error(
        y_true,
        y_pred,
    )


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    X_valid=None,
    y_valid=None,
):

    metrics = {}

    _add_regression_metrics(
        metrics,
        'train',
        y_train,
        model.predict(X_train),
    )

    if X_valid is not None and y_valid is not None:
        _add_regression_metrics(
            metrics,
            'valid',
            y_valid,
            model.predict(X_valid),
        )

    _add_regression_metrics(
        metrics,
        'test',
        y_test,
        model.predict(X_test),
    )

    return metrics


def plot_predictions(y_test, y_pred, fd_name):

    plt.figure(figsize=(10, 4))

    plt.plot(
        y_test,
        label='Actual RUL',
    )

    plt.plot(
        y_pred,
        label='Predicted RUL',
    )

    plt.title(f'{fd_name} Predictions')

    plt.legend()

    plt.show()


def get_feature_importance(
    model,
    feature_names,
    top_k=10,
):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    )

    return (
        importance
        .sort_values(ascending=False)
        .head(top_k)
    )
