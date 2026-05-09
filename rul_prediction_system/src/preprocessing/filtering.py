import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

class SensorFilter:
    def __init__(self):
        self.columns_to_drop = []

    def fit(self, train_df, fd_name):
        # 1. DANH SÁCH "TỬ THẦN" BẮT BUỘC: cycle luôn phải bỏ khỏi feature
        cols_to_drop = ['cycle'] 

        # 2. Xử lý Operational Settings theo từng loại dataset
        op_cols = [f'op_setting_{i}' for i in range(1, 4)]
        multi_condition = ['FD002', 'FD004']

        if fd_name not in multi_condition:
            # Ở FD001/003, op_settings là hằng số -> Xóa sạch
            cols_to_drop.extend(op_cols)
            print(f"   -> [SensorFilter] FD001/003 detected: Dropping all op_settings")
        else:
            # Ở FD002/004, giữ lại cả 3 op_settings (như bạn muốn)
            print(f"   -> [SensorFilter] FD002/004 detected: Keeping all 3 op_settings")

        # 3. Lọc Sensor dựa trên độ biến thiên (Variance)
        feature_cols = [col for col in train_df.columns if col.startswith('sensor_')]
        std_vals = train_df[feature_cols].std()

        # Cột có độ biến thiên bằng 0 (Hằng số tuyệt đối)
        constant_cols = std_vals[np.isclose(std_vals, 0)].index.tolist()
        cols_to_drop.extend(constant_cols)

        # 4. Lọc Sensor "gần hằng số" dựa trên tương quan Spearman (Sức khỏe động cơ)
        near_constant_cols = std_vals[(std_vals > 0) & (std_vals < 0.01)].index.tolist()

        for col in near_constant_cols:
            engine_corrs = []
            # Kiểm tra tương quan trên từng Unit ID để đảm bảo tính khách quan
            for _, group in train_df.groupby('unit_id'):
                if len(group) < 10: continue # Tăng lên 10 để corr chính xác hơn
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    corr, _ = spearmanr(group[col], group['RUL'])

                if not pd.isna(corr):
                    engine_corrs.append(abs(corr))

            avg_corr = np.mean(engine_corrs) if engine_corrs else 0
            
            # Nếu sensor thay đổi cực ít và không ăn nhập gì với RUL -> Xóa
            if avg_corr < 0.2:
                cols_to_drop.append(col)

        # Lưu danh sách cuối cùng (loại bỏ trùng lặp)
        self.columns_to_drop = list(set(cols_to_drop))
        print(f"   -> [SensorFilter] Final drop list ({len(self.columns_to_drop)} cols): {self.columns_to_drop}")

    def transform(self, df):
        # errors='ignore' để tránh lỗi nếu cột đã bị drop trước đó
        return df.drop(columns=self.columns_to_drop, errors='ignore')