# training/trainer.py
import mlflow
import numpy as np
import keras
import logging

# from numpy import load_csv_to_numpy
from mlflow.models import infer_signature
# from models.autoencoder import create_contractive_autoencoder
from .metrics import compute_rmse
from .thresholding import choose_optimal_threshold


def train_model(
    model: keras.Model,
    train_df: np.ndarray,
    test_df: np.ndarray,
    epochs: int = 10,
    batch_size: int = 80
) -> keras.Model:
    
    """Обучает модель автокодировщика на нормальных данных."""
    history = model.fit(
        train_df, 
        train_df,
        validation_data = (test_df, test_df),
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,
        verbose = 1 )

    return model, history.history


def compare_weights(model1, model2, tolerance=1e-5):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    """
    Сравнивает веса двух моделей
    """
    
    if len(weights1) != len(weights2):
        print("Модели имеют разное количество слоев с весами")
        return False
    
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        if not np.allclose(w1, w2, rtol=tolerance, atol=tolerance):
            print(f"Различие в весах на слое {i}")
            return False
        
    return True