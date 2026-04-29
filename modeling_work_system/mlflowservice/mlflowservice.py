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


