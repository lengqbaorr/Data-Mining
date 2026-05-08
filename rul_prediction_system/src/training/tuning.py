from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


XGB_PARAM_GRID = {
    'n_estimators': [500, 800],
    'learning_rate': [0.01, 0.03],
    'max_depth': [3],
    'min_child_weight': [15, 20, 30],
    'gamma': [10, 15, 20],
    'subsample': [0.5, 0.6],
    'colsample_bytree': [0.5, 0.6],
    'reg_alpha': [5, 10, 20],
    'reg_lambda': [50, 100, 200]
}


def tune_xgb(
    X_train,
    y_train,
    groups,
    gkf
):

    model = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        verbosity=0
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=XGB_PARAM_GRID,
        n_iter=30,
        cv=gkf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train,
        groups=groups
    )

    return search.best_params_