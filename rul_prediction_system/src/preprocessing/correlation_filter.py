import numpy as np


class CorrelationFilter:

    def __init__(self, threshold=0.95):

        self.threshold = threshold
        self.to_drop = []

    def fit(self, train_df):

        feature_cols = [
            c for c in train_df.columns
            if c not in ['unit_id', 'cycle', 'RUL']
        ]

        corr_matrix = train_df[feature_cols].corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

    def transform(self, df):

        return df.drop(columns=self.to_drop, errors='ignore')