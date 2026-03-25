import keras
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from .basedetector import BaseAnomalyDetector



class AutoEncoder(BaseAnomalyDetector):
    
    def __init__(self):
        pass

    def get_model(input_dim: int = 26) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(10, activation='elu'),
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])
        model.compile(optimizer="adam", loss="mse", metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])
        return model



