
def add_piecewise_rul(train_df, max_rul=125):

    max_cycle_df = (
        train_df.groupby('unit_id')['cycle']
        .max()
        .reset_index()
    )

    max_cycle_df.columns = ['unit_id', 'max_cycle']

    train_df = train_df.merge(max_cycle_df, on='unit_id', how='left')

    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

    train_df['RUL'] = train_df['RUL'].clip(upper=max_rul)

    train_df.drop(columns=['max_cycle'], inplace=True)

    return train_df