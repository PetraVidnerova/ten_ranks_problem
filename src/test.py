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
    model, epochs = create_model(name, input_shape=(125,1), classes=2)
    model.summary()
    
    model.compile(#loss="mse",
        loss='binary_crossentropy',
        #        optimizer=RMSprop(lr=0.000005, decay=1e-07),
        optimizer=RMSprop(),
        #                  optimizer=Adam(lr=0.00001),
        metrics=['accuracy'])
 

    X = X.reshape(-1, 125, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=42)


    model.fit(X_train, Y_train,
              batch_size=516,
              epochs=5000,
              validation_data=(X_test, Y_test),
              verbose=2)
#              shuffle=True)

    # for layer in model.layers:
    #     print(f"{layer.name}: {layer.trainable}") 

    # for layer in model.layers:
    #     if "dropout" not in layer.name:
    #         layer.trainable = True

    # for layer in model.layers:
    #     print(f"{layer.name}: {layer.trainable}") 
        
#     model.compile(#loss="mse",
#         loss='binary_crossentropy',
#         #        optimizer=RMSprop(lr=0.000005, decay=1e-07),
#         optimizer=RMSprop(),
#         #                  optimizer=Adam(lr=0.00001),
#         metrics=['accuracy'])

        
#     model.fit(X_train, Y_train,
#               batch_size=2**16,
#               epochs=10000,
#               validation_data=(X_test, Y_test),
#               verbose=2)
# #              shuffle=True)


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

    import sys
    class_num = int(sys.argv[1])

    
    X = np.load(f"{DATA_DICT}/X_0vs{class_num}.npy")
    y = np.load(f"{DATA_DICT}/y_0vs{class_num}.npy")

    
    model = 1

    test_model(MODELS[model], X, y)
    # for name in MODELS:
    #     test_model(name, X, y)
