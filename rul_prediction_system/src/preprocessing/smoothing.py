class EMASmoother:

    def __init__(self, span=10):
        self.span = span

    def transform(self, df):

        sensor_cols = [
            c for c in df.columns
            if c.startswith('sensor_')
        ]

        df[sensor_cols] = df.groupby('unit_id')[sensor_cols].transform(
            lambda x: x.ewm(span=self.span, adjust=False).mean()
        )

        return df