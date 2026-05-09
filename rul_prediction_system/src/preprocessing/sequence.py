import numpy as np
import pandas as pd

class SequenceGenerator:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length

    def _gen_sequence(self, id_df, seq_cols):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(
            range(0, num_elements - self.sequence_length + 1), 
            range(self.sequence_length, num_elements + 1)
        ):
            yield data_matrix[start:stop, :]

    def _gen_labels(self, id_df, label_cols):
        data_matrix = id_df[label_cols].values
        num_elements = data_matrix.shape[0]
        return data_matrix[self.sequence_length - 1:num_elements, :]

    def generate_train_sequences(self, train_df, feature_cols):
        """Tạo mảng 3D cho tập Train"""
        X_train = np.concatenate(list(
            (list(self._gen_sequence(train_df[train_df['unit_id'] == unit_id], feature_cols)) 
             for unit_id in train_df['unit_id'].unique())
        )).astype(np.float32)
        
        y_train = np.concatenate([
            self._gen_labels(train_df[train_df['unit_id'] == unit_id], ['RUL']) 
            for unit_id in train_df['unit_id'].unique()
        ]).astype(np.float32)
        
        return X_train, y_train

    def generate_test_last_windows(self, test_df, feature_cols, rul_truth):
        """Tạo mảng 3D (chỉ cửa sổ cuối) cho tập Test kèm Padding"""
        X_test_last = []
        y_test_last = []
        
        unit_ids = test_df['unit_id'].unique()
        
        for i, unit_id in enumerate(unit_ids):
            unit_data = test_df[test_df['unit_id'] == unit_id]
            
            if len(unit_data) >= self.sequence_length:
                last_window = unit_data[feature_cols].values[-self.sequence_length:, :]
                X_test_last.append(last_window)
            else:
                # Dùng padding mode='edge' như code gốc của bạn
                padding_len = self.sequence_length - len(unit_data)
                last_window = unit_data[feature_cols].values
                padded_window = np.pad(last_window, ((padding_len, 0), (0, 0)), mode='edge')
                X_test_last.append(padded_window)
                
            y_test_last.append(rul_truth[i])
            
        X_test_last = np.array(X_test_last).astype(np.float32)
        y_test_last = np.array(y_test_last).astype(np.float32).reshape(-1, 1)
        
        return X_test_last, y_test_last