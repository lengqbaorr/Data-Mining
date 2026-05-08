
import pandas as pd

from src.data.loader import load_dataset
from src.inference.predictor import RULPredictor


# ==========================================================
# LOAD ORIGINAL TEST DATA
# ==========================================================

dataset = load_dataset(
    data_dir="CMaps",
    fd_name="FD001"
)

test_df = dataset["test"]
rul_df = dataset["rul"]

print("Original test dataset loaded.")
print(f"Test shape: {test_df.shape}")


# ==========================================================
# TAKE 1 ENGINE CASE
# ==========================================================

UNIT_ID = 1

raw_df = test_df[
    test_df["unit_id"] == UNIT_ID
].copy()

print(f"\nSelected unit_id = {UNIT_ID}")
print(raw_df.head())


# ==========================================================
# GROUND TRUTH RUL
# ==========================================================

true_rul = rul_df.iloc[UNIT_ID - 1]["RUL"]

print(f"\nGround Truth RUL: {true_rul}")


# ==========================================================
# CREATE PREDICTOR
# ==========================================================

predictor = RULPredictor(
    artifact_dir="artifacts/FD001",
    fd_name="FD001"
)


# ==========================================================
# PREDICT
# ==========================================================

preds = predictor.predict(raw_df)

print("\nPrediction Result:")
print(preds)


# ==========================================================
# COMPARE
# ==========================================================

predicted_rul = preds["predicted_rul"].iloc[0]

error = abs(predicted_rul - true_rul)

print("\nComparison:")
print(f"Predicted RUL : {predicted_rul:.2f}")
print(f"Actual RUL    : {true_rul:.2f}")
print(f"Absolute Error: {error:.2f}")
