import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from model import create_model, MODELS
from preload_data import DATA_DICT
from tensorflow.keras.layers import Reshape 

import autokeras as ak



def test_model(X, y):


#    X = X.reshape(-1, 124)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=42)

    # model = ak.AutoModel(
    #     inputs=[ak.Input()],
    #     outputs=[ak.ClassificationHead()],
    #     max_trials=100
    # )
    
    model = ak.StructuredDataClassifier(max_trials=30, overwrite=True)
    model.fit(X_train, Y_train,
              batch_size=2**10,
              epochs=1000,
              verbose=2)
    results = model.predict(X_test)


    _, train_acc = model.evaluate(X_train, Y_train, verbose=2)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    _, acc = model.evaluate(X, y, verbose=2)
    
    print(f"Train accuracy:  {train_acc}")
    print(f"Test accuracy:  {test_acc}")
    print(f"Total accuracy:  {acc}")

    
    # y_pred = np.argmax(model.predict(X_test), axis=1)
    # y_test = np.argmax(Y_test, axis=1)
    y_pred = np.around(model.predict(X))
    y_true = y
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    print(classification_report(y_true, y_pred))
    
if __name__ == "__main__":

    # X = np.load(f"{DATA_DICT}/X_3vs6.npy")
    # y = np.load(f"{DATA_DICT}/y_3vs6.npy")

    X = np.load(f"{DATA_DICT}/X_0vs1.npy")
    y = np.load(f"{DATA_DICT}/y_0vs1.npy")


    test_model(X, y)
    # for name in MODELS:
    #     test_model(name, X, y)
