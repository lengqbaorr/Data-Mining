import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

class SensorFilter:

    def __init__(self):
        self.columns_to_drop = []

    def fit(self, train_df, fd_name):

        sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
        op_cols = [f'op_setting_{i}' for i in range(1, 4)]

        multi_condition = ['FD002', 'FD004']

        cols_to_drop = []

        if fd_name not in multi_condition:
            cols_to_drop.extend(op_cols)

        feature_cols = [
            col for col in train_df.columns
            if col.startswith('sensor_') or col.startswith('op_setting_')
        ]

        std_vals = train_df[feature_cols].std()

        constant_cols = std_vals[
            np.isclose(std_vals, 0)
        ].index.tolist()

        cols_to_drop.extend(constant_cols)

        near_constant_cols = std_vals[
            (std_vals > 0) & (std_vals < 0.01)
        ].index.tolist()

        for col in near_constant_cols:

            engine_corrs = []

            for _, group in train_df.groupby('unit_id'):

                if len(group) < 5:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)

                    corr, _ = spearmanr(
                        group[col],
                        group['RUL']
                    )

                if not pd.isna(corr):
                    engine_corrs.append(abs(corr))

            avg_corr = np.mean(engine_corrs) if engine_corrs else 0

            if avg_corr < 0.2:
                cols_to_drop.append(col)

        self.columns_to_drop = list(set(cols_to_drop))

    def transform(self, df):
        return df.drop(columns=self.columns_to_drop, errors='ignore')