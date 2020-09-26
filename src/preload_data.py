import numpy as np
import pandas as pd

DATA_DICT = "../data/tens_rank_8_1e4"
MAX_CLASS = 8
N = 126


X = []

target = None
for class_number in range(1, MAX_CLASS+1):
    df = pd.read_csv(f"{DATA_DICT}/tens_rank_0{class_number}.txt", sep=" ", header=None)
    if len(df.columns) != N:
        raise ValueError("inconsistent data")
    df[0] -= 1
    X.append(df.to_numpy())


X = np.concatenate(X)
# shuffle
np.random.shuffle(X)

# label
y = X[:, 0]
X = X[:, 1:]

np.save(f"{DATA_DICT}/X", X)
np.save(f"{DATA_DICT}/y", y)

print("Saved.")
