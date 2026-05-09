from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DLScaler:
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_columns = None

    def fit_transform(self, df):
        df_scaled = df.copy()
        
        # Chỉ scale các cột sensor và op_settings
        exclude_cols = ['unit_id', 'cycle', 'RUL']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        df_scaled[self.feature_columns] = self.scaler.fit_transform(df_scaled[self.feature_columns])
        return df_scaled

    def transform(self, df):
        df_scaled = df.copy()
        # Transform dựa trên những cột đã lưu ở bước fit
        df_scaled[self.feature_columns] = self.scaler.transform(df_scaled[self.feature_columns])
        return df_scaled