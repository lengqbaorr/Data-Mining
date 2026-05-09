from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def group_train_valid_split(
    X,
    y,
    groups,
    valid_size=0.2,
    random_state=42,
):
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=valid_size,
        random_state=random_state,
    )

    train_idx, valid_idx = next(
        gss.split(X, y, groups)
    )

    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]

    X_val = X.iloc[valid_idx]
    y_val = y.iloc[valid_idx]

    split = {
        'X_tr': X_tr,
        'y_tr': y_tr,
        'X_val': X_val,
        'y_val': y_val,
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'valid_size': valid_size,
        'random_state': random_state,
    }

    return split


def train_final_model(
    X_train,
    y_train,
    groups,
    best_params,
    valid_size=0.2,
    random_state=42,
):
    split = group_train_valid_split(
        X_train,
        y_train,
        groups,
        valid_size=valid_size,
        random_state=random_state,
    )

    model = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=25,
        **best_params,
    )

    model.fit(
        split['X_tr'],
        split['y_tr'],
        eval_set=[(split['X_val'], split['y_val'])],
        verbose=False,
    )

    return model, split


def train_final_rf_model(
    X_train,
    y_train,
    groups,
    best_params,
    valid_size=0.2,
    random_state=42,
):
    split = group_train_valid_split(
        X_train,
        y_train,
        groups,
        valid_size=valid_size,
        random_state=random_state,
    )

    rf_model = RandomForestRegressor(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
        warm_start=True 
    )

    best_val_rmse = float('inf')
    best_n_estimators = 0
    stop_count = 0
    patience = 5  
    step_trees = 50 
    
    print("\n[RF Training] Bắt đầu huấn luyện với Manual Early Stopping trên tập Valid...")
    
    for i in range(1, 21): 
        current_trees = i * step_trees
        rf_model.n_estimators = current_trees
        
        rf_model.fit(split['X_tr'], split['y_tr'])
        
        val_preds = rf_model.predict(split['X_val'])
        current_rmse = np.sqrt(mean_squared_error(split['y_val'], val_preds))
        
        if current_rmse < best_val_rmse:
            best_val_rmse = current_rmse
            best_n_estimators = current_trees
            stop_count = 0
        else:
            stop_count += 1
            
        if stop_count >= patience:
            print(f">>> Early stopping kích hoạt tại {current_trees} cây (Valid RMSE tốt nhất: {best_val_rmse:.3f}).")
            break
            
    print(f"[RF Training] Chốt model với số cây: {best_n_estimators}")
    rf_model.warm_start = False 
    rf_model.n_estimators = best_n_estimators if best_n_estimators > 0 else step_trees
    rf_model.fit(split['X_tr'], split['y_tr']) 

    return rf_model, split