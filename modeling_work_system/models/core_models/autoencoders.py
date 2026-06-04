import tensorflow
import keras
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from ...metrics.aemetrics import AEMetricResult
from ...metrics.metrics import ExperimentMetric
from typing import Dict, Any, Optional


STANDART_AE = keras.Sequential([
    keras.layers.Dense(26, activation='relu', input_shape=(26,)),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(26, activation='linear')
])

COMPACT_AE = keras.Sequential([
    keras.layers.Dense(26, activation='relu', input_shape=(26,)),
    keras.layers.Dense(11, activation='relu'),
    keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(11, activation='relu'),
    keras.layers.Dense(26, activation='linear')
])

STANDART_AE.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
COMPACT_AE.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')