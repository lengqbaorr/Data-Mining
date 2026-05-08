import requests
import pandas as pd

from src.data.loader import load_dataset


# ==========================================================
# LOAD SAMPLE
# ==========================================================

dataset = load_dataset(
    data_dir="CMaps",
    fd_name="FD001"
)

test_df = dataset["test"]

sample = test_df[
    test_df["unit_id"] == 1
]


# ==========================================================
# CALL API
# ==========================================================

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "data": sample.to_dict(orient="records")
    }
)

print(response.json())
