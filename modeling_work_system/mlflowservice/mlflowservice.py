import mlflow
import mlflow.keras
import dagshub
import numpy as np
import keras
import logging
import pandas as pd
import tempfile
import os
import json

# from numpy import load_csv_to_numpy
from ..models.autoencoders.basedetector_interface import BaseAnomalyDetector
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    accuracy_score, 
    confusion_matrix
)


class Mlflowservice:
    def __init__(
            self,
            mlflow_tracking_uri: str, 
            mlflow_repo_owner: str, 
            mlflow_repo_name: str, 
            mlflow_username: str,
            mlflow_pass: str,
            mlflow_token: str):
        
        # Иницианилзируем данные для подключения к dugshub
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
        logging.info("--- DUGSHUB-MLFLOW CONFIGURATION COMPLETE ---")

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_repo_owner = mlflow_repo_owner
        self.mlflow_repo_name = mlflow_repo_name
        self.mlflow_username = mlflow_username
        self.mlflow_pass = mlflow_pass
        self.mlflow_token = mlflow_token

        # self.train_data = train_data
        
        # self.epochs = epochs
        # self.batch_size = batch_size

        # self.model_type = model_type
        # self.model_name = model_name
        # self.experiment_name = experiment_name
        logging.info("--- EXPERIMENT CONFIGURATION COMPLETE ---")

        return None


#======================================================
    def load_model_from_mlflow(
        self,
        model_name: str = "test_model",
        # experiment_name: str = "Autoencoder_Anomaly_v2",
        stage: str = "None"  # или "Staging", "None", либо конкретная версия как строка "1"
        ) -> keras.Model:
        
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

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Формируем URI модели в формате MLflow
        model_uri = f"models:/{model_name}/latest"

        try:
            model = mlflow.keras.load_model(
                model_uri 
                # compile=False
                )
            logging.info(f"The model is loaded from mlflow: {model_uri}")
            return model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow by URI '{model_uri}': {e}")


# ======================================================
    def save_model_to_mlflow(
        self,
        model: BaseAnomalyDetector,

        training_history: dict,
        threshold: dict,
        epochs,
        batch_size,

        
        model_name: str = "test_model",
        experiment_name: str = "Autoencoder_Anomaly_v2",
        feature_names: list = None,

        additional_params: dict = None,
        log_predictions: bool = False,
        max_samples_log: int = 100
    ):
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            
            # ==================== ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ====================
            mlflow.log_param("model_type", "Autoencoder")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # ======================================================
            # ================== МЕТРИКИ ОБУЧЕНИЯ ==================
            # ======================================================
            # История по эпохам
            if training_history is not None:
                for epoch, (loss, val_loss) in enumerate(
                    zip(training_history.get("loss", []), training_history.get("val_loss", []))
                ):
                    mlflow.log_metric("train_loss", float(loss), step=epoch)
                    mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                
                # Финальные потери
                if training_history.get("loss"):
                    mlflow.log_metric("final_train_loss", float(training_history["loss"][-1]))
                    mlflow.log_metric("final_val_loss", float(training_history.get("val_loss", [-1])[-1]))
            
            # ======================================================
            # ============= МЕТРИКИ ПОРОГА (VALIDATION) ============
            # ======================================================
            mlflow.log_metric("optimal_threshold", threshold)

            # ======================================================
            # ======================= МОДЕЛЬ =======================
            # ======================================================
            # Работет тоько для моделей  [keras|sclearn]
            mlflow.keras.log_model(
                model=model.get_model_core(),
                artifact_path="model",
                registered_model_name=model_name,
                # signature=signature
                # input_example=X_sample[:1]  # Пример входа для Model Registry
            )

            return run.info.run_id
