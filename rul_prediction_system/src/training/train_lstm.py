import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.loader import load_dataset
from src.data.labeling import add_piecewise_rul
from src.preprocessing.pipeline import CMAPSSPreprocessor
from src.preprocessing.sequence import SequenceGenerator

from src.utils.config import load_config
from src.utils.io import save_pickle, save_json

# =========================================================
# 1. CONFIGURATION
# =========================================================
DATA_DIR = 'CMaps'
FD_NAME = 'FD003' # Đổi thành FD002, FD003, FD004 tùy ý
SEQUENCE_LENGTH = 30 # Độ dài cửa sổ thời gian

print(f"[{FD_NAME}] BẮT ĐẦU PIPELINE HUẤN LUYỆN LSTM...")
config = load_config(f'configs/{FD_NAME.lower()}.yaml')

# =========================================================
# 2. DATA LOADING & LABELING
# =========================================================
dataset = load_dataset(DATA_DIR, FD_NAME)
train_df = dataset['train']
test_df = dataset['test']
rul_truth = dataset['rul']['RUL'].values # Ground truth cho tập Test

train_df = add_piecewise_rul(train_df, max_rul=config['max_rul'])

# =========================================================
# 3. CHIA TẬP TRAIN / VALID (80/20) TRƯỚC KHI TIỀN XỬ LÝ
# =========================================================
print(f"\n[Data Split] Đang chia Train/Valid (80/20) theo Unit_ID...")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, valid_idx = next(gss.split(train_df, groups=train_df['unit_id']))

# Tách dữ liệu THÔ ra làm 2 phần
train_split_raw = train_df.iloc[train_idx].copy()
valid_split_raw = train_df.iloc[valid_idx].copy()

# =========================================================
# 4. PREPROCESSING (Chỉ FIT trên 80% Train thực sự)
# =========================================================
print("\n[Preprocessing] Đang xử lý dữ liệu (Chế độ DL: Smoothing & MinMax Scale)...")
preprocessor = CMAPSSPreprocessor(config)

# Cực kỳ quan trọng: Lệnh FIT bây giờ CHỈ nhìn thấy 80% dữ liệu Train
train_processed = preprocessor.fit_transform(train_split_raw, FD_NAME, model_type='dl')

# Các tập còn lại chỉ được TRANSFORM (dùng ké Min/Max của tập Train)
valid_processed = preprocessor.transform(valid_split_raw, FD_NAME, model_type='dl')
test_processed = preprocessor.transform(test_df, FD_NAME, model_type='dl')

feature_cols = preprocessor.feature_columns

# =========================================================
# 5. SEQUENCE GENERATION (WINDOWING)
# =========================================================
print(f"\n[Sequence Generation] Đang tạo cửa sổ 3D (Window Size = {SEQUENCE_LENGTH})...")
seq_gen = SequenceGenerator(sequence_length=SEQUENCE_LENGTH)

# Tạo mảng 3D từ các tập đã xử lý độc lập
X_train, y_train = seq_gen.generate_train_sequences(train_processed, feature_cols)
X_valid, y_valid = seq_gen.generate_train_sequences(valid_processed, feature_cols)
X_test, y_test = seq_gen.generate_test_last_windows(test_processed, feature_cols, rul_truth)

print(f" -> Train shape: X={X_train.shape}, y={y_train.shape}")
print(f" -> Valid shape: X={X_valid.shape}, y={y_valid.shape}")
print(f" -> Test shape : X={X_test.shape}, y={y_test.shape}")

# =========================================================
# 6. LSTM MODEL ARCHITECTURE
# =========================================================
print("\n[Model Building] Khởi tạo kiến trúc LSTM...")
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1) # Dự báo RUL
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# =========================================================
# 7. TRAINING
# =========================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

print("\n[Training] Bắt đầu huấn luyện...")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_valid, y_valid), # Sử dụng tập Valid đã tách sạch sẽ
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

# =========================================================
# 8. EVALUATION
# =========================================================
print("\n[Evaluation] Đang tính toán Metrics...")
y_pred_train = model.predict(X_train, verbose=0)
y_pred_valid = model.predict(X_valid, verbose=0)
y_pred_test = model.predict(X_test, verbose=0)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)

rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
mae_valid = mean_absolute_error(y_valid, y_pred_valid)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

metrics = {
    "train_rmse": float(rmse_train),
    "train_mae": float(mae_train),
    "valid_rmse": float(rmse_valid),
    "valid_mae": float(mae_valid),
    "test_rmse": float(rmse_test),
    "test_mae": float(mae_test),
    "training_time_seconds": float(training_time)
}

print(f" -> Train RMSE: {rmse_train:.4f} | Train MAE: {mae_train:.4f}")
print(f" -> Valid RMSE: {rmse_valid:.4f} | Valid MAE: {mae_valid:.4f}")
print(f" -> Test RMSE : {rmse_test:.4f} | Test MAE : {mae_test:.4f}")

# =========================================================
# 9. SAVE ARTIFACTS
# =========================================================
artifact_dir = f'artifacts/{FD_NAME}'
os.makedirs(artifact_dir, exist_ok=True)

# Lưu Model Keras (hỗ trợ đuôi .h5 hoặc định dạng Keras mới)
model.save(f'{artifact_dir}/lstm_model.h5')

# Lưu Preprocessor (chứa DLScaler đã fit)
save_pickle(preprocessor, f'{artifact_dir}/dl_preprocessor.pkl')

# Lưu cấu hình sequence và feature columns
save_json({
    "sequence_length": SEQUENCE_LENGTH,
    "feature_columns": feature_cols
}, f'{artifact_dir}/dl_config.json')

# Lưu Metrics
save_json(metrics, f'{artifact_dir}/lstm_metrics.json')

print(f"\n✅ Đã lưu toàn bộ Artifacts LSTM vào '{artifact_dir}/'")