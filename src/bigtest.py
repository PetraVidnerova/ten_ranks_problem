import time 
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from model import create_model, MODELS
from preload_data import DATA_DICT


def test_model(name, X, y):
    print("MODEL: ", name)
    model, epochs = create_model(name, input_shape=(1,125), classes=8)
    model.summary()
    
    model.compile(#loss="mse",
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.0001, decay=2e-05),
        #                  optimizer=Adam(lr=0.00001),
        metrics=['accuracy'])


    X = X.reshape(-1, 1, 125)
    Y_b = to_categorical(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_b,
                                                        stratify=y, 
                                                        test_size=0.2,
                                                        random_state=42)


    model.fit(X_train, Y_train,
              batch_size=2**16,
              epochs=epochs,
              validation_data=(X_test, Y_test),
              verbose=2)
#              shuffle=True)


    _, train_acc = model.evaluate(X_train, Y_train, verbose=2)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=2)

    print(f"Train accuracy:  {train_acc}")
    print(f"Test accuracy:  {test_acc}")

    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(Y_test, axis=1)
    #y_test = Y_test
    #    print(y_pred)
    #    print(y_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))

    print("Saving ... ", end=" ")
    model.save(f"ten_ranks_{name}_{time.time()}")
    print("saved") 

if __name__ == "__main__":

    X = np.load(f"{DATA_DICT}/X.npy")
    y = np.load(f"{DATA_DICT}/y.npy")

    import sys
    model = int(sys.argv[1])
    
    test_model(MODELS[model], X, y)
    # for name in MODELS:
    #     test_model(name, X, y)
