from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

XGB_PARAM_GRID = {
    'n_estimators': [500, 800],           # Giảm số lượng cây để tránh học quá sâu
    'learning_rate': [0.01, 0.03],        # Tốc độ học rất chậm để tìm điểm hội tụ mượt
    'max_depth': [3],                     # Ép cây cực nông (chỉ 3 lớp) để chống Overfit
    'min_child_weight': [15, 20, 30],     # Tăng mạnh: Nhánh phải có rất nhiều dữ liệu mới được mọc
    'gamma': [10, 15, 20],                # Tăng thuế "mọc nhánh" cực cao
    'subsample': [0.5, 0.6],              # Chỉ dùng 50-60% dữ liệu cho mỗi cây (tăng tính ngẫu nhiên)
    'colsample_bytree': [0.5, 0.6],       # Chỉ dùng 50-60% số cột cho mỗi cây
    'reg_alpha': [5, 10, 20],             # L1 Regularization: Triệt tiêu các feature yếu
    'reg_lambda': [50, 100, 200]          # L2 Regularization: Phạt cực nặng các trọng số lớn
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Bộ tham số cực kỳ chặt chẽ chống Overfit của bạn
RF_PARAM_GRID_CUSTOM = {
    'max_depth': [5, 7, 8],                
    'min_samples_leaf': [150, 200, 300],   
    'max_features': [0.2, 0.3],            
    'min_samples_split': [300, 400],       
    'max_samples': [0.5, 0.6],             
    'bootstrap': [True]
}

def tune_rf(X_train, y_train, groups, gkf):
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    search = GridSearchCV(
        estimator=model,
        param_grid=RF_PARAM_GRID_CUSTOM,
        cv=gkf, # Dùng GroupKFold thay vì cv=3 ngẫu nhiên
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    search.fit(
        X_train, 
        y_train, 
        groups=groups
    )
    
    return search.best_params_