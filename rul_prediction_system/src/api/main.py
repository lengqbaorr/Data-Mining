from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import traceback

from src.inference.predictor import RULPredictor

app = FastAPI()

predictor = RULPredictor(
    artifact_dir="artifacts/FD001",
    fd_name="FD001"
)


class PredictRequest(BaseModel):
    data: list


@app.post("/predict")
def predict(request: PredictRequest):

    try:

        df = pd.DataFrame(request.data)

        preds = predictor.predict(df)

        return {
            "predicted_rul": preds.to_dict(orient="records")
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "error": str(e)
        }