import warnings
import os
from pandas.errors import PerformanceWarning
from scipy.stats import ConstantInputWarning
from sklearn.model_selection import GroupKFold

from src.data.loader import load_dataset
from src.data.labeling import add_piecewise_rul
from src.preprocessing.pipeline import CMAPSSPreprocessor

from src.training.tuning import tune_rf 
from src.training.trainer import train_final_rf_model
from src.training.evaluate import (
    evaluate_model,
    plot_predictions,
    get_feature_importance
)

from src.utils.config import load_config
from src.utils.io import (
    save_pickle,
    save_json
)

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = 'CMaps'
FD_NAME = 'FD002' # Thay đổi tùy ý: FD001, FD002, FD003, FD004

config = load_config(f'configs/{FD_NAME.lower()}.yaml')

# =========================================================
# LOAD DATA & LABELING
# =========================================================
dataset = load_dataset(DATA_DIR, FD_NAME)
train_df = dataset['train']
test_df = dataset['test']
rul_df = dataset['rul']

train_df = add_piecewise_rul(train_df, max_rul=config['max_rul'])

# =========================================================
# PREPROCESSING
# =========================================================
preprocessor = CMAPSSPreprocessor(config)
train_processed = preprocessor.fit_transform(train_df, FD_NAME)
test_processed = preprocessor.transform(test_df, FD_NAME)

# =========================================================
# BUILD TRAIN/TEST DATA
# =========================================================
X_train = train_processed[preprocessor.feature_columns]
y_train = train_processed['RUL']
groups = train_processed['unit_id']

test_last = test_processed.groupby('unit_id').last().reset_index()
X_test = test_last[preprocessor.feature_columns]
y_test = rul_df['RUL'].values

# =========================================================
# HYPERPARAMETER TUNING (RANDOM FOREST)
# =========================================================
gkf = GroupKFold(n_splits=5)

# Gọi hàm tune cho RF
best_params = tune_rf(
    X_train,
    y_train,
    groups,
    gkf
)

print('\nBest Parameters (Random Forest):')
print(best_params)

# =========================================================
# FINAL TRAINING (RANDOM FOREST)
# =========================================================
model, split = train_final_rf_model(
    X_train,
    y_train,
    groups,
    best_params,
    valid_size=0.2,
)

# =========================================================
# EVALUATION 
# =========================================================
metrics = evaluate_model(
    model,
    split['X_tr'],
    split['y_tr'],
    X_test,
    y_test,
    X_valid=split['X_val'],
    y_valid=split['y_val'],
)

print('\nMetrics (Random Forest):')
print(metrics)

# =========================================================
# FEATURE IMPORTANCE & PLOT
# =========================================================
importance = get_feature_importance(model, preprocessor.feature_columns)
print('\nTop Features:')
print(importance)

y_pred = model.predict(X_test)
plot_predictions(y_test, y_pred, f'{FD_NAME} - Random Forest')

# =========================================================
# SAVE ARTIFACTS
# =========================================================
artifact_dir = f'artifacts/{FD_NAME}'
os.makedirs(artifact_dir, exist_ok=True)

save_pickle(model, f'{artifact_dir}/rf_model.pkl')
# Không cần lưu lại preprocessor và feature_columns nếu chạy XGB trước đó đã lưu
# nhưng lưu đè cũng không ảnh hưởng gì, đảm bảo pipeline độc lập.
save_pickle(preprocessor, f'{artifact_dir}/preprocessor.pkl')
save_json(preprocessor.feature_columns, f'{artifact_dir}/feature_columns.json')
save_json(metrics, f'{artifact_dir}/rf_metrics.json')

print('\nRandom Forest Artifacts saved successfully.')