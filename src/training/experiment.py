import mlflow
import dagshub
import numpy as np
import keras
import logging
import pandas as pd
import tempfile
import os

# from numpy import load_csv_to_numpy
from mlflow.models import infer_signature
# from models.autoencoder import create_contractive_autoencoder
from .metrics import compute_rmse
from .trainer import train_model
from .thresholding import choose_optimal_threshold


class Experiment:

    """
    Класс для произведения экспериментов
    Поддерживает методы для проведения эксперимента, что включает в себя:
     1. Обучение модели
     2. Сохранение логов в mlflow
     3. Выгрузку модели из mlflow
    """

    def __init__(
            self,
            mlflow_tracking_uri: str, 
            mlflow_repo_owner: str, 
            mlflow_repo_name: str, 
            mlflow_username: str,
            mlflow_pass: str,
            mlflow_token: str
            ):
    
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_pass
        os.environ['MLFLOW_TRACKING_TOKEN'] = mlflow_token

        dagshub.auth.add_app_token(token = mlflow_token)
        dagshub.init(
            repo_owner = mlflow_repo_owner, 
            repo_name = mlflow_repo_name, 
            mlflow = True
            )
        
        # dagshub.mlflow.set_tracking_uri(mlflow_tracking_uri)
        logging.info("КОНФИГУРАЦИЯ DUGSHUB-MLFLOW ЗАВЕРШЕНА")

        return None

# ======================================================
    def send_experiment_to_mlflow(
        self,
        model: keras.Model,
        training_history,

        X_train: np.ndarray,
        X_test: np.ndarray,
        X_val: np.ndarray,
        X_anomaly: np.ndarray,

        threshold: float,
        threshold_accuracy: float,
        df_threshold_results: pd.DataFrame,

        experiment_name: str,
        registered_model_name: str,

        epochs: int,
        batch_size: int
        ):
        """
        Логирует уже ОБУЧЕННУЮ модель и связанные метрики в MLflow.
        Вычисляет метрики самостоятельно.
        """

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:

            # Кастомные метрики
            X_recon_val = model.predict(X_test)
            rmse_val = float(keras.metrics.RootMeanSquaredError()(X_test, X_recon_val).numpy())

            mlflow.log_metric("rmse_validation", rmse_val)
            mlflow.log_metric("anomaly_threshold", threshold)
            mlflow.log_metric("anomaly_detection_accuracy", threshold_accuracy)

            # Параметры
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("input_dim", X_train.shape[1])

            # История обучения по эпохам
            for epoch, (loss, val_loss) in enumerate(
                zip(training_history["loss"], training_history["val_loss"])
            ):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            # Модель
            signature = infer_signature(X_train, model.predict(X_train[:min(10, len(X_train))]))
            mlflow.keras.log_model(
                model,
                artifact_path = "autoencoder",
                registered_model_name = registered_model_name,
                signature = signature
            )

            # Артефакт: результаты подбора порога
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = os.path.join(tmp_dir, "threshold_analysis.csv")
                df_threshold_results.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path)

            return run.info.run_id

# ======================================================
    def load_model_from_mlflow(
        self,
        registered_model_name: str,
        stage: str = "None",  # или "Staging", "None", либо конкретная версия как строка "1"
        tracking_uri: str = None ) -> keras.Model:
        
        """
        Загружает модель из MLflow Model Registry.
        
        Parameters
        ----------
        registered_model_name : str
            Имя модели в MLflow Registry (например, "autoencoder_turbo").
        stage : str, optional
            Стадия модели: "Production", "Staging", "None" (последняя версия),
            или номер версии в виде строки, например "3".
        tracking_uri : str, optional
            URI для подключения к MLflow (например, "file:///path/to/mlruns" или "http://localhost:5000").
            Если не указан — используется текущий активный URI.
        
        Returns
        -------
        mlflow.pyfunc.PyFuncModel
            Загруженная модель, готовая к вызову через `.predict(X)`.
        """

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Формируем URI модели в формате MLflow
        model_uri = f"models:/{registered_model_name}/{stage}"

        try:
            model = mlflow.keras.load_model(model_uri)
            logging.info(f"Модель загружена из mlflow: {model_uri}")
            return model
        
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель из MLflow по URI '{model_uri}': {e}")