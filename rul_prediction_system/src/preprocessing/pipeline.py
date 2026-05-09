from src.preprocessing.filtering import SensorFilter
from src.preprocessing.normalization import Normalizer
from src.preprocessing.smoothing import EMASmoother
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.correlation_filter import CorrelationFilter
from src.preprocessing.scaler import DLScaler
from src.preprocessing.label_engineering import PiecewiseRUL 

class CMAPSSPreprocessor:
    def __init__(self, config):
        self.config = config
        
        # KHỞI TẠO CÁC MODULE
        self.sensor_filter = SensorFilter()
        self.normalizer = Normalizer(num_regimes=config['num_regimes'])
        self.smoother = EMASmoother(span=config['ema_span'])
        self.label_engineer = PiecewiseRUL(max_rul=config.get('max_rul', 125))
        
        # NHÁNH ML
        self.feature_engineer = FeatureEngineer(
            window_size=config['window_size'],
            shift_size=config['shift_size']
        )
        self.correlation_filter = CorrelationFilter(threshold=config['corr_threshold'])
        
        # NHÁNH DL
        self.dl_scaler = DLScaler()
        self.feature_columns = None

    def fit_transform(self, train_df, fd_name, model_type='ml'):
        print(f"\n[Pipeline] Bắt đầu fit_transform cho {fd_name} ({model_type.upper()})")

        # BƯỚC 1: Lọc cảm biến cơ bản (QUAN TRỌNG: Phải gọi fit trước)
        self.sensor_filter.fit(train_df, fd_name)
        train_df = self.sensor_filter.transform(train_df)

        # BƯỚC 2: Xử lý nhãn RUL (Piecewise)
        train_df = self.label_engineer.transform(train_df)

        # BƯỚC 3: RẼ NHÁNH XỬ LÝ
        if model_type == 'dl':
            # Nhánh DL: Smooth -> Drop Hardcode -> Scale
            train_df = self.smoother.transform(train_df)
            
            # Loại bỏ các cột hằng số bổ sung hoặc cycle nếu còn sót
            drop_hard = ['sensor_1', 'sensor_5', 'sensor_18', 'sensor_19', 'cycle']
            train_df = train_df.drop(columns=[c for c in drop_hard if c in train_df.columns], errors='ignore')
            
            # Xác định danh sách feature thực tế cho LSTM
            self.feature_columns = [
                c for c in train_df.columns 
                if c not in ['unit_id', 'RUL', 'regime', 'cycle']
            ]
            
            # Ép DataFrame chỉ lấy đúng những cột này để tránh "kẻ mạo danh"
            train_df = train_df[['unit_id', 'RUL'] + self.feature_columns]
            
            print(f"   -> [DL] Số lượng features cuối cùng: {len(self.feature_columns)}")
            print(f"   -> [DL] Danh sách: {self.feature_columns}")

            # Scale dữ liệu
            train_df = self.dl_scaler.fit_transform(train_df)
            
        else:
            # --- NHÁNH ML ---
            self.normalizer.fit(train_df, fd_name)
            train_df = self.normalizer.transform(train_df, fd_name)
            train_df = self.smoother.transform(train_df)
            
            # ÉP XÓA VẬT LÝ CYCLE TRƯỚC KHI VÀO FEATURE ENGINEER
            train_df = train_df.drop(columns=['cycle'], errors='ignore') # Cực kỳ quan trọng
            
            train_df = self.feature_engineer.transform(train_df)
            
            self.correlation_filter.fit(train_df)
            train_df = self.correlation_filter.transform(train_df)
            
            # Lấy danh sách feature cuối cùng
            self.feature_columns = [
                c for c in train_df.columns 
                if c not in ['unit_id', 'RUL', 'regime', 'cycle']
            ]
            
            # CHỐT HẠ: In ra số lượng để kiểm tra
            print(f"   -> [ML] Số lượng features cuối cùng: {len(self.feature_columns)}")
        return train_df

    def transform(self, df, fd_name, model_type='ml'):
        # Tạo bản sao để tránh thay đổi dữ liệu gốc ngoài ý muốn
        df = df.copy()
        
        # 1. Transform các module đã fit
        df = self.sensor_filter.transform(df)
        df = self.label_engineer.transform(df)

        if model_type == 'dl':
            # 2. Nhánh DL: Smooth
            df = self.smoother.transform(df)
            
            # 3. Drop các cột rác
            drop_hard = ['sensor_1', 'sensor_5', 'sensor_18', 'sensor_19', 'cycle']
            df = df.drop(columns=[c for c in drop_hard if c in df.columns], errors='ignore')
            
            # 4. ÉP DATAFRAME (Sửa lỗi KeyError tại đây)
            # Kiểm tra xem 'RUL' có trong index không trước khi lấy
            cols_to_keep = ['unit_id']
            if 'RUL' in df.columns:
                cols_to_keep.append('RUL')
            
            # Chỉ lấy các feature đã được xác định khi fit tập Train
            # (Đảm bảo shape của Test luôn khớp với Train)
            df = df[cols_to_keep + self.feature_columns]
            
            # 5. Scale
            df = self.dl_scaler.transform(df)
            
        else:
            # Nhánh ML
            df = self.normalizer.transform(df, fd_name)
            df = self.smoother.transform(df)
            df = self.feature_engineer.transform(df)
            df = self.correlation_filter.transform(df)

        return df