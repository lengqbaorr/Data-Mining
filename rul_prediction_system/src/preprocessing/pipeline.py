from src.preprocessing.filtering import SensorFilter
from src.preprocessing.normalization import Normalizer
from src.preprocessing.smoothing import EMASmoother
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.correlation_filter import CorrelationFilter


class CMAPSSPreprocessor:

    def __init__(self, config):

        self.config = config

        self.sensor_filter = SensorFilter()

        self.normalizer = Normalizer(
            num_regimes=config['num_regimes']
        )

        self.smoother = EMASmoother(
            span=config['ema_span']
        )

        self.feature_engineer = FeatureEngineer(
            window_size=config['window_size'],
            shift_size=config['shift_size']
        )

        self.correlation_filter = CorrelationFilter(
            threshold=config['corr_threshold']
        )

        self.feature_columns = None

    def fit_transform(self, train_df, fd_name):

        self.sensor_filter.fit(train_df, fd_name)
        train_df = self.sensor_filter.transform(train_df)

        self.normalizer.fit(train_df, fd_name)
        train_df = self.normalizer.transform(train_df, fd_name)

        train_df = self.smoother.transform(train_df)

        train_df = self.feature_engineer.transform(train_df)

        self.correlation_filter.fit(train_df)
        train_df = self.correlation_filter.transform(train_df)

        self.feature_columns = [
            c for c in train_df.columns
            if c not in ['unit_id', 'cycle', 'RUL']
        ]

        return train_df

    def transform(self, df, fd_name):

        df = self.sensor_filter.transform(df)

        df = self.normalizer.transform(df, fd_name)

        df = self.smoother.transform(df)

        df = self.feature_engineer.transform(df)

        df = self.correlation_filter.transform(df)

        return df