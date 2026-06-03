import tensorflow
import keras
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from ...metrics.aemetrics import AEMetricResult
from ...metrics.metrics import ExperimentMetric
from typing import Dict, Any, Optional


STANDART_AE = keras.Sequential([
    keras.layers.Dense(26, activation='elu', input_shape=(26,)),
    keras.layers.Dense(16, activation='elu'),
    keras.layers.Dense(10, activation='elu'),
    keras.layers.Dense(16, activation='elu'),
    keras.layers.Dense(26, activation='elu')
])

EXPANSION_AE = keras.Sequential([
    keras.layers.Dense(26, activation='elu', input_shape=(26,)),
    keras.layers.Dense(28, activation='elu'),
    keras.layers.Dense(30, activation='elu'),
    keras.layers.Dense(28, activation='elu'),
    keras.layers.Dense(26, activation='elu')
])

STANDART_AE.compile(
            optimizer="adam", 
            loss="mse", 
            metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])

EXPANSION_AE.compile(
            optimizer="adam", 
            loss="mse", 
            metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])