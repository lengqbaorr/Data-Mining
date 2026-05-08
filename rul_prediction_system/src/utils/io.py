import joblib
import json
import os


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)