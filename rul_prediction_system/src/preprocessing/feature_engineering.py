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
        if 'cycle' in df_feat.columns:
            df_feat = df_feat.drop(columns=['cycle'])


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

        # 4. Hợp nhất: Chỉ giữ lại 'unit_id', 'RUL' và các cột mới tính
        # Loại bỏ các cột sensor gốc nếu bạn muốn nhánh ML chỉ dùng feature trích xuất
        # Hoặc ít nhất là loại bỏ 'cycle'
        cols_to_keep = [c for c in df_feat.columns if c in ['unit_id', 'RUL']]
        
        final_df = pd.concat(
            [df_feat[cols_to_keep], feature_df], 
            axis=1
        )

        return final_df.astype('float32', errors='ignore')