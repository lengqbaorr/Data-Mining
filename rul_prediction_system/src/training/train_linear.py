import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.loader import load_dataset
from src.data.labeling import add_piecewise_rul
from src.preprocessing.pipeline import CMAPSSPreprocessor
from src.utils.config import load_config
from src.utils.io import save_pickle, save_json

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = "CMaps"
FD_NAME = "FD001"

print("="*60)
print(f"[{FD_NAME}] BẮT ĐẦU PIPELINE HUẤN LUYỆN LINEAR REGRESSION")
print("="*60)

config = load_config(f"configs/{FD_NAME.lower()}.yaml")

# =========================================================
# 1. LOAD DỮ LIỆU
# =========================================================
print("\n[1] Đang tải dữ liệu gốc...")
dataset = load_dataset(DATA_DIR, FD_NAME)

train_df = dataset["train"]
test_df = dataset["test"]
rul_df = dataset["rul"]

# =========================================================
# 2. GÁN NHÃN (LABELING)
# =========================================================
print(f"\n[2] Đang gán nhãn Piecewise RUL (max_rul={config.get('max_rul', 125)})...")
train_df = add_piecewise_rul(train_df, max_rul=config.get("max_rul", 125))

# =========================================================
# 3. TIỀN XỬ LÝ (PREPROCESSING - HƯỚNG ML)
# =========================================================
print("\n[3] Đang chạy Pipeline Tiền xử lý (ML mode)...")
preprocessor = CMAPSSPreprocessor(config)

train_processed = preprocessor.fit_transform(train_df, FD_NAME, model_type="ml")
test_processed = preprocessor.transform(test_df, FD_NAME, model_type="ml")

# =========================================================
# 4. CHUẨN BỊ TẬP TRAIN & TEST THỰC TẾ
# =========================================================
features = preprocessor.feature_columns

X_train = train_processed[features].values
y_train = train_processed["RUL"].values
groups = train_processed["unit_id"].values

# CMAPSS yêu cầu đánh giá RUL ở chu kỳ cuối cùng của mỗi động cơ trong tập Test
test_last = test_processed.groupby("unit_id").last().reset_index()
X_test = test_last[features].values
y_test = rul_df["RUL"].values

print(f"\n[4] Trích xuất dữ liệu hoàn tất:")
print(f"    -> Số lượng features : {len(features)}")
print(f"    -> Kích thước X_train: {X_train.shape}")
print(f"    -> Kích thước X_test : {X_test.shape}")

# =========================================================
# 5. CROSS VALIDATION (GROUP K-FOLD)
# =========================================================
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

best_model = None
best_score = float("inf")
fold_rmses = []

print(f"\n[5] Bắt đầu Cross Validation ({n_splits} Folds) - Mô hình: Ridge Regression...")
print("-" * 60)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Tính tỉ lệ phần trăm trực tiếp để in ra log
    train_pct = len(X_tr) / len(X_train) * 100
    val_pct = len(X_val) / len(X_train) * 100

    print(f" Fold {fold + 1}:")
    print(f"    + Train: {len(X_tr):>5} mẫu ({train_pct:.1f}%) | Valid: {len(X_val):>5} mẫu ({val_pct:.1f}%)")

    # Khởi tạo mô hình tuyến tính có điều chuẩn (Ridge)
    model = Ridge(alpha=1.0) 
    model.fit(X_tr, y_tr)

    # Đánh giá trên tập Validation
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    fold_rmses.append(rmse)

    print(f"    -> Valid RMSE: {rmse:.4f}")

    if rmse < best_score:
        best_score = rmse
        best_model = model

print("-" * 60)
print(f"[CV Summary] Mean RMSE : {np.mean(fold_rmses):.4f} (± {np.std(fold_rmses):.4f})")
print(f"             Best RMSE : {best_score:.4f} (Được chọn làm Final Model)")

# =========================================================
# 6. ĐÁNH GIÁ TRÊN TẬP TEST GỐC (GROUND TRUTH)
# =========================================================
print("\n[6] Đang đánh giá Final Model trên tập Test (Ground Truth)...")
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

metrics = {
    "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
    "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
    "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
}

print("\n[KẾT QUẢ METRICS CHÍNH THỨC]")
print(f"  + Train RMSE : {metrics['train_rmse']:>7.4f}")
print(f"  + Train MAE  : {metrics['train_mae']:>7.4f}")
print(f"  + TEST RMSE  : {metrics['test_rmse']:>7.4f}  <-- Điểm quan trọng nhất")
print(f"  + TEST MAE   : {metrics['test_mae']:>7.4f}")

# =========================================================
# 7. LƯU ARTIFACTS
# =========================================================
artifact_dir = f"artifacts/{FD_NAME}"
os.makedirs(artifact_dir, exist_ok=True)

save_pickle(best_model, f"{artifact_dir}/linear_model.pkl")
save_pickle(preprocessor, f"{artifact_dir}/linear_preprocessor.pkl")

# Ensure feature_columns is a list before saving to JSON
features_list = list(features) if not isinstance(features, list) else features
save_json(features_list, f"{artifact_dir}/linear_feature_columns.json")
save_json(metrics, f"{artifact_dir}/linear_metrics.json")

print(f"\n[7] Đã lưu Artifacts thành công tại thư mục: {artifact_dir}")
print("="*60)