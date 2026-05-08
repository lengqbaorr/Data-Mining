from src.data.loader import load_dataset


dataset = load_dataset(
    data_dir="CMaps",
    fd_name="FD001"
)


test_df = dataset['test']

unit_df = test_df[
    test_df['unit_id'] == 1
]

unit_df.to_csv(
    "sample_engine.csv",
    index=False
)