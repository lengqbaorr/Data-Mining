import numpy as np
import pandas as pd

class PiecewiseRUL:
    def __init__(self, max_rul=125):
        self.max_rul = max_rul

    def fit(self, df=None):
        # Piecewise RUL thường cố định ngưỡng, không cần học từ dữ liệu
        return self

    def transform(self, df):
        """
        Áp dụng ngưỡng max_rul cho cột RUL nếu nó tồn tại
        """
        if 'RUL' in df.columns:
            df = df.copy()
            # Giới hạn giá trị RUL tối đa
            df['RUL'] = df['RUL'].clip(upper=self.max_rul)
        return df