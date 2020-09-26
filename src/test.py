import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from model import create_model, MODELS
from preload_data import DATA_DICT


def test_model(name, X, y):
    print("MODEL: ", name)
    model, epochs = create_model(name, input_shape=(125,), classes=8)

    model.compile(optimizer='adam',
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)

    model.fit(X_train, y_train, epochs=epochs)

    train_acc = model.evaluate(X_train, y_train, verbose=2)
    test_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"Train accuracy:  {train_acc}")
    print(f"Test accuracy:  {train_acc}")


if __name__ == "__main__":

    X = np.load(f"{DATA_DICT}/X.npy")
    y = np.load(f"{DATA_DICT}/y.npy")
    y_b = to_categorical(y)

    test_model("simple_dense_model", X, y_b)
    # for name in MODELS:
    #     test_model(name, X, y)
