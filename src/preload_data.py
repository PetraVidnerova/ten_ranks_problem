import numpy as np
import pandas as pd

DATA_DICT = "../data/tens_rank_8_1e4"
MAX_CLASS = 8
N = 126


def load_data(valid_columns=None, label=None):
    X = []

    if valid_columns is None:
        valid_columns = list(range(1, MAX_CLASS+1))

    target = None
    for i, class_number in enumerate(valid_columns):
        df = pd.read_csv(f"{DATA_DICT}/tens_rank_0{class_number}.txt", sep=" ", header=None)
        if len(df.columns) != N:
            raise ValueError("inconsistent data")
        df[0] = i
        X.append(df.to_numpy())

    X = np.concatenate(X)
    # shuffle
    np.random.shuffle(X)

    # label
    y = X[:, 0]
    X = X[:, 1:]

    if label is None:
        label = ""
    else:
        label = "_" + label

    np.save(f"{DATA_DICT}/X{label}", X)
    np.save(f"{DATA_DICT}/y{label}", y)

    print("Saved.")


if __name__ == "__main__":
    load_data([2, 5], label="2vs5")
