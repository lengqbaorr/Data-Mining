import pandas as pd
import os


COL_NAMES = (
    ['unit_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)


def load_dataset(data_dir, fd_name):

    train_path = os.path.join(data_dir, f'train_{fd_name}.txt')
    test_path = os.path.join(data_dir, f'test_{fd_name}.txt')
    rul_path = os.path.join(data_dir, f'RUL_{fd_name}.txt')

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COL_NAMES)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COL_NAMES)
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    return {
        'train': train_df,
        'test': test_df,
        'rul': rul_df
    }