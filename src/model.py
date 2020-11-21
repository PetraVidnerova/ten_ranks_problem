from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Reshape, Conv3D, AveragePooling3D
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, Model


def _simple_dense_model(input_shape, classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(classes)
    ])
    return model


def _dense_model_2(input_shape, classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid'),
        #Dense(classes, activation="softmax")
    ])
    return model


def _simple_conv_network(input_shape, classes):
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, padding='same',
               activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(classes)
    ])
    return model


def _conv_network2(input_shape, classes):
    model = Sequential([
        Reshape((1, 125), input_shape=input_shape),
        Conv1D(filters=16, kernel_size=3, padding='same',
               activation='relu', input_shape=input_shape),
        Dropout(0.25),
        #        MaxPool1D(pool_size=2),
        Conv1D(filters=8, kernel_size=3, padding='same',
               activation='relu'),
        Dropout(0.25),
        Flatten(),
#        Dense(32, activation='relu'),        
        Dense(16, activation='relu'),
        Dense(classes)
    ])
    return model

def _conv_network3D(input_shape, classes):
    model = Sequential([
        Reshape((1,5,5,5), input_shape=input_shape),
        Conv3D(filters=64, kernel_size=3, padding='same',
               activation='relu'),
#        AveragePooling3D(pool_size=2),
        Dropout(0.25),
        Conv3D(filters=32, kernel_size=3, padding='same',
               activation='relu'),
        Dropout(0.25),
        Flatten(),        
        Dense(16, activation='relu'),
        Dense(classes, activation="softmax")
    ])
    return model

def _conv_network3D_2(input_shape, classes):
    print(" HULA HOP ")
    model = Sequential([
        Reshape((1,5,5,5), input_shape=input_shape),
        Conv3D(filters=32, kernel_size=3, padding='same',
               activation='relu'),
#        AveragePooling3D(pool_size=2),
        Dropout(0.25),
        Conv3D(filters=16, kernel_size=3, padding='same',
               activation='relu'),
        Dropout(0.25),
        Flatten(),        
        Dense(32, activation='relu'),
#        Dense(, activation='relu'),
#        Dense(classes, activation="softmax")
        Dense(1, activation = 'sigmoid')
    ])
    return model

def _saved(index, input_shape, classes):
    model = load_model("saved_model")

    dense_layer = Dense(8, activation='relu', name="dense_layer_new")(model.layers[-3].output)
    #    dropout_layer = Dropout(0.2, name="new_dropout")(dense_layer)
    dropout_layer = dense_layer
    if index == 2:
        dense_layer2 = Dense(4, activation='relu', name="dense_layer_new2")(dropout_layer)
    pre_output = dropout_layer if index == 1 else dense_layer2
    output_layer = Dense(1, activation='sigmoid', name="output_sigmoid_layer")(pre_output)
    
    model2 = Model(inputs=model.input, outputs=[output_layer])
    model2.summary()

    for layer in model.layers[:-2]:
        layer.trainable = False
        
    for layer in model.layers:
        print(layer.name, layer.trainable)
    
    return model2


_saved2 = partial(_saved, 2)
_saved1 = partial(_saved, 1)




MODELS = [
    "simple_dense_model", #0
    "dense_model_2",
    "simple_conv_network",
    "conv_network2",
    "conv_network3D",  # 4
    "conv_network3D_2", # 5
    "saved2",
    "saved1"
]

EPOCHS = {
    "simple_dense_model": 100,
    "dense_model_2": 10000,
    "simple_conv_network": 100,
    "conv_network2": 10000,
    "conv_network3D": 50000,
    "conv_network3D_2": 50000,
    "saved2": 50000,
    "saved1": 50000
}


def create_model(name, input_shape, classes):
    return globals()[f"_{name}"](input_shape, classes), EPOCHS[name]
