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


# def train_model(
#     train_df: np.ndarray,
#     test_df: np.ndarray,
#     valid_df: np.ndarray,
#     anomal_df: np.ndarray,

#     experiment_name: str,
#     registered_model_name: str,
    
#     model:  keras.Model,
#     epochs: int = 10,
#     batch_size: int = 80
#     ):
    
#     # Настройка MLflow
#     mlflow.set_experiment(experiment_name)
#     mlflow.keras.autolog()

#     with mlflow.start_run() as run:
#         # Обучение
#         history = model.fit(
#             train_df, train_df,
#             validation_data = (test_df, test_df),
#             epochs = epochs,
#             batch_size = batch_size,
#             shuffle = True
#         )

#         # Предсказание
#         X_recon = model.predict(valid_df)

#         # Метрики
#         rmse = compute_rmse(valid_df, X_recon)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_param("epochs", epochs)

#         # Логирование модели
#         signature = infer_signature(train_df, model.predict(train_df[:10]))
#         mlflow.keras.log_model(
#             model,
#             artifact_path = "autoencoder",
#             registered_model_name = registered_model_name,
#             signature = signature
#         )


#         optimal_line, optimal_df = choose_optimal_threshold(
#             model = model,
#             normal_control_df = valid_df,
#             anomaly_control_df = anomal_df,
#             )

#         logging.info(optimal_df)

#     # run_id = run.info.run_id
#     return model, optimal_df



def train_model(
    model: keras.Model,
    train_df: np.ndarray,
    test_df: np.ndarray,
    epochs: int = 10,
    batch_size: int = 80
) -> keras.Model:
    
    """Обучает модель автокодировщика на нормальных данных."""
    model.fit(
        train_df, train_df,
        validation_data = (test_df, test_df),
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,
        verbose = 1 )

    return model


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