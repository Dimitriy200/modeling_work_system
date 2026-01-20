import keras
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError


def create_sparse_autoencoder(input_dim: int = 26) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
        keras.layers.Dense(30, activation='elu'),
        keras.layers.Dense(36, activation='elu'),
        keras.layers.Dense(30, activation='elu'),
        keras.layers.Dense(input_dim, activation='elu')
    ])
    model.compile(optimizer="adam", loss="mse", metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])
    return model