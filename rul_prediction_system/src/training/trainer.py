from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


def train_final_model(
    X_train,
    y_train,
    groups,
    best_params
):

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, valid_idx = next(
        gss.split(X_train, y_train, groups)
    )

    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train.iloc[valid_idx]
    y_val = y_train.iloc[valid_idx]

    model = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=25,
        **best_params
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model