import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras import Sequential


def _simple_dense_model(input_shape, classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(classes)
    ])
    return model

    return model


def _dense_model_2(input_shape, classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(classes)
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
        Conv1D(filters=32, kernel_size=3, padding='same',
               activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(classes)
    ])
    return model


MODELS = [
    "simple_dense_model",
    "dense_model_2",
    "simple_conv_network",
    "conv_network2"
]

EPOCHS = {
    "simple_dense_model": 10,
    "dense_model_2": 10,
    "simple_conv_network": 10,
    "conv_network2": 10
}


def create_model(name, input_shape, classes):
    return globals()[f"_{name}"](input_shape, classes), EPOCHS[name]
