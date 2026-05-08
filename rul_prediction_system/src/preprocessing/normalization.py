import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class Normalizer:

    def __init__(self, num_regimes=6):

        self.num_regimes = num_regimes

        self.scaler = None
        self.op_scaler = None
        self.kmeans = None
        self.regime_scalers = {}

    def fit(self, train_df, fd_name):

        sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
        if fd_name in ['FD002', 'FD004']:

            op_cols = [f'op_setting_{i}' for i in range(1, 4)]

            self.op_scaler = StandardScaler()

            op_scaled = self.op_scaler.fit_transform(train_df[op_cols])

            self.kmeans = KMeans(
                n_clusters=self.num_regimes,
                random_state=42,
                n_init=10
            )

            train_df['regime'] = self.kmeans.fit_predict(op_scaled)

            for regime in range(self.num_regimes):

                idx = train_df[train_df['regime'] == regime].index

                if len(idx) == 0:
                    continue

                scaler = StandardScaler()

                scaler.fit(train_df.loc[idx, sensor_cols])

                self.regime_scalers[regime] = scaler

            train_df.drop(columns=['regime'], inplace=True)

        else:

            self.scaler = StandardScaler()
            self.scaler.fit(train_df[sensor_cols])

    def transform(self, df, fd_name):

        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]

        if fd_name in ['FD002', 'FD004']:

            op_cols = [f'op_setting_{i}' for i in range(1, 4)]

            op_scaled = self.op_scaler.transform(df[op_cols])

            df['regime'] = self.kmeans.predict(op_scaled)

            for regime, scaler in self.regime_scalers.items():

                idx = df[df['regime'] == regime].index

                if len(idx) == 0:
                    continue

                df.loc[idx, sensor_cols] = scaler.transform(
                    df.loc[idx, sensor_cols]
                )

            df.drop(columns=['regime'], inplace=True)

        else:

            df[sensor_cols] = self.scaler.transform(df[sensor_cols])

        return df