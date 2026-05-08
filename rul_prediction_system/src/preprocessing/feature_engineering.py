import pandas as pd


class FeatureEngineer:

    def __init__(
        self,
        window_size=15,
        shift_size=10
    ):

        self.window_size = window_size
        self.shift_size = shift_size

    def transform(self, df):

        df_feat = df.copy()

        sensor_cols = [
            c for c in df.columns
            if c.startswith('sensor_')
        ]

        grouped = df_feat.groupby('unit_id')

        feature_dict = {}

        for col in sensor_cols:

            col_group = grouped[col]

            # ==========================================
            # ROLLING MEAN
            # ==========================================

            feature_dict[f'{col}_roll_mean'] = (
                col_group
                .rolling(self.window_size, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # ==========================================
            # ROLLING STD
            # ==========================================

            feature_dict[f'{col}_roll_std'] = (
                col_group
                .rolling(self.window_size, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

            # ==========================================
            # TREND
            # ==========================================

            feature_dict[f'{col}_trend'] = (
                df_feat[col]
                - col_group.shift(self.shift_size)
            ).fillna(0)

            # ==========================================
            # LAG1
            # ==========================================

            feature_dict[f'{col}_lag1'] = (
                col_group
                .shift(1)
                .fillna(df_feat[col])
            )

            # ==========================================
            # CUMAX
            # ==========================================

            feature_dict[f'{col}_cumax'] = (
                col_group.cummax()
            )

        # ==========================================
        # CONCAT ONCE
        # ==========================================

        feature_df = pd.DataFrame(
            feature_dict,
            index=df_feat.index
        )

        df_feat = pd.concat(
            [df_feat, feature_df],
            axis=1
        )

        return df_feat.astype('float32', errors='ignore')