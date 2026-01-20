import keras
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError


def create_contractive_autoencoder(input_dim: int = 26) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
        keras.layers.Dense(16, activation='elu', activity_regularizer="l1"),
        keras.layers.Dense(10, activation='elu', activity_regularizer="l1"),
        keras.layers.Dense(16, activation='elu', activity_regularizer="l1"),
        keras.layers.Dense(input_dim, activation='elu', activity_regularizer="l1")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])
    return model