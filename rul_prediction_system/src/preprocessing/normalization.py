import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class Normalizer:
    def __init__(self, num_regimes=6):
        self.num_regimes = num_regimes
        self.scaler = None
        self.op_scaler = None
        self.kmeans = None
        self.regime_scalers = {}
        self.valid_sensors = None  # Danh sách sensor "sống" sau khi xử lý

    def fit(self, train_df, fd_name):
        # Tạo bản sao và ép kiểu float64 để tính toán chính xác nhất
        temp_df = train_df.copy()
        sensor_cols = [c for c in temp_df.columns if c.startswith('sensor_')]
        temp_df[sensor_cols] = temp_df[sensor_cols].astype(np.float64)
        
        if fd_name in ['FD002', 'FD004']:
            op_cols = [f'op_setting_{i}' for i in range(1, 4)]
            temp_df[op_cols] = temp_df[op_cols].astype(np.float64)
            
            # 1. Fit bộ scaler cho các thông số vận hành (Operational Settings)
            self.op_scaler = StandardScaler()
            op_scaled = self.op_scaler.fit_transform(temp_df[op_cols])

            # 2. Phân nhóm điều kiện bay (Clustering Regimes)
            self.kmeans = KMeans(
                n_clusters=self.num_regimes,
                random_state=42,
                n_init=10
            )
            temp_df['regime'] = self.kmeans.fit_predict(op_scaled)

            # 3. Chuẩn hóa cảm biến theo từng nhóm (Regime-wise Normalization)
            for regime in range(self.num_regimes):
                idx = temp_df[temp_df['regime'] == regime].index
                if len(idx) == 0:
                    continue

                scaler = StandardScaler()
                # Fit trên dữ liệu của nhóm hiện tại
                scaler.fit(temp_df.loc[idx, sensor_cols])
                self.regime_scalers[regime] = scaler
                
                # Cập nhật giá trị để kiểm tra độ lệch chuẩn sau chuẩn hóa
                temp_df.loc[idx, sensor_cols] = scaler.transform(temp_df.loc[idx, sensor_cols])

            # --- DYNAMIC SENSOR FILTERING ---
            # Tính độ lệch chuẩn của toàn bộ sensor sau khi đã normalize theo nhóm
            stds = temp_df[sensor_cols].std()
            
            # Chỉ giữ các sensor thực sự thay đổi (std > 0)
            self.valid_sensors = stds[stds > 1e-5].index.tolist()
            
            dropped = list(set(sensor_cols) - set(self.valid_sensors))
            if dropped:
                print(f"   -> [Normalizer] FD002/004 phát hiện {len(dropped)} sensor rác: {dropped}")
        
        else:
            # Đối với FD001/003: Chỉ cần lọc sensor hằng số và Standardize
            stds = temp_df[sensor_cols].std()
            self.valid_sensors = stds[stds > 1e-5].index.tolist()
            
            self.scaler = StandardScaler()
            self.scaler.fit(temp_df[self.valid_sensors])

    def transform(self, df, fd_name):
        df = df.copy()
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        
        # Ép kiểu float32 ngay để tránh lỗi 'LossySetitem' khi gán giá trị float vào cột int
        df[sensor_cols] = df[sensor_cols].astype(np.float32)

        if fd_name in ['FD002', 'FD004']:
            op_cols = [f'op_setting_{i}' for i in range(1, 4)]
            
            # Ép kiểu float64 để làm hài lòng C-backend của Scikit-Learn
            op_data = df[op_cols].astype(np.float64)
            op_scaled = self.op_scaler.transform(op_data)
            
            # Dự đoán Regime (Sử dụng np.ascontiguousarray để tránh lỗi Buffer mismatch)
            df['regime'] = self.kmeans.predict(np.ascontiguousarray(op_scaled))

            for regime, scaler in self.regime_scalers.items():
                idx = df[df['regime'] == regime].index
                if len(idx) == 0:
                    continue

                # Lấy dữ liệu sensor của nhóm này
                s_data = df.loc[idx, sensor_cols].astype(np.float64)
                scaled = scaler.transform(s_data)
                
                # Gán lại kết quả với kiểu float32 để tối ưu bộ nhớ cho Deep Learning
                df.loc[idx, sensor_cols] = scaled.astype(np.float32)

            df.drop(columns=['regime'], inplace=True)
        else:
            # Xử lý đơn giản cho FD001/003
            s_data = df[self.valid_sensors].astype(np.float64)
            df[self.valid_sensors] = self.scaler.transform(s_data).astype(np.float32)

        # Trả về các cột ID, Cycle... và chỉ các sensor hợp lệ
        other_cols = [c for c in df.columns if not c.startswith('sensor_')]
        return df[other_cols + self.valid_sensors]