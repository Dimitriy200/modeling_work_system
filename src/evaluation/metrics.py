# evaluation/metrics.py
import numpy as np
import keras

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse_metric = keras.metrics.RootMeanSquaredError()
    rmse_metric.update_state(y_true, y_pred)
    return float(rmse_metric.result().numpy())

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return keras.losses.mean_squared_error(y_true, y_pred).numpy()