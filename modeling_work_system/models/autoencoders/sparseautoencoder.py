import keras
import numpy as np
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from .basedetector_interface import BaseAnomalyDetector
from typing import Dict, Any


class SparseAutoEncoder(BaseAnomalyDetector):
    
    def __init__(self):
        self.model: keras.Model = None
        self.history = None
    
# ======================================================
    def build_model(self, input_dim: int=26)->keras.Model:
        model = keras.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
            keras.layers.Dense(30, activation='elu'),
            keras.layers.Dense(50, activation='elu'),
            keras.layers.Dense(30, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])

        model.compile(
            optimizer="adam", 
            loss="mse", 
            metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])
        
        return model

# ======================================================
    def fit(self, 
            X_train: np.ndarray,
            X_test: np.ndarray,
            epochs: int=5,
            batch_size: int=80)->Dict[keras.Model, Any]:
        
        self.model = self.build_model()
        self.history = self.model.fit(
            X_train, 
            X_train,
            validation_data=(X_test, X_test),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1)
        
        train_result = {
            "model": self.model,
            "history":self.history.history
        }

        return train_result

# ======================================================
    def predict(self, X):
        return self.model.predict(X)