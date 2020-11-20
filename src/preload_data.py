import numpy as np
import pandas as pd

DATA_DICT = "../data/tens_rank_6_1e5"
MAX_CLASS = 6
N = 126


def load_data(valid_columns=None, label=None):
    X = []

    if valid_columns is None:
        valid_columns = list(range(1, MAX_CLASS+1))

    target = None
    for i, class_name in enumerate(valid_columns):
        df = pd.read_csv(f"{DATA_DICT}/tens_{class_name}.txt", sep=" ", header=None)
        if len(df.columns) != N:
            raise ValueError("inconsistent data")
        df[0] = i
        X.append(df.to_numpy())
        print(i)

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
    for i in range(1, MAX_CLASS+1):        
        load_data(["unif", f"rank_0{i}"], label=f"0vs{i}")
