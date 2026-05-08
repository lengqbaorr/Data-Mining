import pandas as pd

from src.utils.io import load_pickle


class RULPredictor:

    def __init__(self, artifact_dir, fd_name):

        self.fd_name = fd_name

        self.model = load_pickle(
            f'{artifact_dir}/xgb_model.pkl'
        )

        self.preprocessor = load_pickle(
            f'{artifact_dir}/preprocessor.pkl'
        )


    def predict(self, raw_df):

        processed = self.preprocessor.transform(
            raw_df,
            self.fd_name
        )

        last_df = (
            processed
            .groupby('unit_id')
            .last()
            .reset_index()
        )

        X = last_df[
            self.preprocessor.feature_columns
        ]

        preds = self.model.predict(X)

        results = pd.DataFrame({
            'unit_id': last_df['unit_id'],
            'predicted_rul': preds
        })

        return results

