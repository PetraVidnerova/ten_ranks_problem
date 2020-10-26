import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


from preload_data import DATA_DICT


import sys

name = sys.argv[1] 

model = load_model(name)

X = np.load(f"{DATA_DICT}/X.npy")
y = np.load(f"{DATA_DICT}/y.npy")

Y = to_categorical(y)

_, acc = model.evaluate(X, Y, verbose=2)

print(f"Accuracy:  {acc}")


y_pred = np.argmax(model.predict(X), axis=1)
y_true = np.argmax(Y, axis=1)
    
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)
print(classification_report(y_true, y_pred))


