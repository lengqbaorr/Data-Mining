import pandas as pd

class FeatureEngineer:
    def __init__(self, window_size=15, shift_size=10):
        self.window_size = window_size
        self.shift_size = shift_size

    def transform(self, df):
        # 1. Tạo bản sao và chủ động chặn 'cycle' để không tạo đặc trưng rác
        df_feat = df.copy()
        if 'cycle' in df_feat.columns:
            df_feat = df_feat.drop(columns=['cycle'])

        sensor_cols = [c for c in df_feat.columns if c.startswith('sensor_')]
        grouped = df_feat.groupby('unit_id')
        feature_dict = {}

        for col in sensor_cols:
            col_group = grouped[col]
            
            # Gom rolling lại để code gọn và chạy nhanh hơn
            roll = col_group.rolling(self.window_size, min_periods=1)

            # ==========================================
            # 1. ROLLING FEATURES (Thống kê cửa sổ trượt)
            # ==========================================
            feature_dict[f'{col}_roll_mean'] = roll.mean().reset_index(level=0, drop=True)
            feature_dict[f'{col}_roll_std']  = roll.std().reset_index(level=0, drop=True).fillna(0)
            feature_dict[f'{col}_roll_min']  = roll.min().reset_index(level=0, drop=True)
            feature_dict[f'{col}_roll_max']  = roll.max().reset_index(level=0, drop=True)

            # ==========================================
            # 2. TREND FEATURE (Biến động dài hạn)
            # ==========================================
            feature_dict[f'{col}_trend'] = (df_feat[col] - col_group.shift(self.shift_size)).fillna(0)

            # ==========================================
            # 3. LAGGING FEATURES (Biến động tức thời)
            # ==========================================
            feature_dict[f'{col}_lag1'] = col_group.shift(1).fillna(df_feat[col])
            feature_dict[f'{col}_lag2'] = col_group.shift(2).fillna(df_feat[col])

            # ==========================================
            # 4. EXPERT FEATURE: CUMULATIVE MAX
            # ==========================================
            feature_dict[f'{col}_cumax'] = col_group.cummax()

        # Tạo DataFrame chứa toàn bộ đặc trưng mới
        feature_df = pd.DataFrame(feature_dict, index=df_feat.index)

        # Hợp nhất DataFrame cũ (chứa op_settings, sensor gốc, RUL...) với features mới
        final_df = pd.concat([df_feat, feature_df], axis=1)

        # Ép kiểu float32 để tiết kiệm RAM và tăng tốc huấn luyện ML
        return final_df.astype('float32', errors='ignore')