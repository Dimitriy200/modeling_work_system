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

from typing import Literal, Optional
# from numpy import load_csv_to_numpy
from ..models.autoencoders.basedetector_interface import BaseAnomalyDetector
from pathlib import Path
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    accuracy_score, 
    confusion_matrix
)


StageVersion = Literal["Production", "Staging", "Archived"]
threshol_name = "optimal_threshold"


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
        stage: StageVersion = "latest"
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
        model_uri = f"models:/{model_name}/{stage}"

        try:
            model = mlflow.keras.load_model(
                model_uri 
                # compile=False
                )
            logging.info(f"The model is loaded from mlflow: {model_uri}")
            
            return model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow by URI '{model_uri}': {e}")


#======================================================
    def load_threshold_from_mlflow(
            self,
            model_name: str = "test_model",
            experiment_name: str = "test_model_run",
            run_id: str = None,
            stage: StageVersion = "latest"):
        """
        Загружает порог (threshold) для указанной модели из MLflow.
        Порог берётся из метрик того run'а, в котором модель была зарегистрирована.
        """
        
        client = MlflowClient()


        # 1 Определяем версию модели
        if stage == "latest":
            # Ищем последнюю версию по номеру (любой стадии)
            versions = client.search_model_versions(
                f"name='{model_name}'",
                order_by=["version_number DESC"],
                max_results=1
            )
            if not versions:
                raise ValueError(f"Модель '{model_name}' не найдена в MLflow Registry")
            model_version = versions[0]
            
        elif stage.isdigit():
            # Если передан номер версии как строка: "3"
            model_version = client.get_model_version(model_name, stage)
            
        else:
            # Если передана стадия: "Production", "Staging", etc.
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"Не найдено версий модели '{model_name}' в стадии '{stage}'")
            model_version = versions[0]
        
        # 2 Получаем run_id, в котором была создана эта версия
        run_id = model_version.run_id
        if not run_id:
            raise RuntimeError(
                f"Не удалось найти run_id для модели {model_name} версии {model_version.version}. "
                "Возможно, модель была зарегистрирована без привязки к run."
            )
        
        # 3 Загружаем порог из метрик этого run'а
        run = mlflow.get_run(run_id)
        threshold = run.data.metrics.get(threshol_name)  # ← убедитесь, что так логируете!
        
        if threshold is None:
            # Пробуем альтернативные имена (если логировали под другим ключом)
            threshold = run.data.metrics.get(threshol_name)
        
        if threshold is None:
            logging.warning(
                f"⚠️ Порог не найден в метриках run {run_id}. "
                f"Доступные метрики: {list(run.data.metrics.keys())}"
            )
            return None
        
        logging.info(
            f"✅ Порог загружен для {model_name} v{model_version.version} "
            f"(stage: {model_version.current_stage}, run: {run_id}): {threshold}"
        )
        return threshold


        # if run_id is None:
        #     run = self._get_run_mlflow(
        #         model_name=model_name,
        #         experiment_name=experiment_name
        #     )
        # else:
        #     run = mlflow.get_run(run_id)

        # # Получение конкретной метрики
        # threshold = run.data.metrics.get("threshold")
        # logging.info(f"Latest threshold is {threshold}")

        # return threshold

#======================================================
    def load_skaller_from_mlflow(
            self,
            # run_id: str,
            model_name: str = "scaller_save",
            # experiment_name: str = "test_model_run",
            versions="latest"):
        
        scaler_uri = f"models:/{model_name}/{versions}"
        scaler = mlflow.sklearn.load_model(scaler_uri)

        return scaler
        

# ======================================================
    def save_model_to_mlflow(
        self,
        model: BaseAnomalyDetector,

        training_history: dict,
        threshold: dict,
        epochs,
        batch_size,

        metrics: dict,

        model_name: str = "test_model",
        experiment_name: str = "Autoencoder_Anomaly_v2",
        stage: Optional[StageVersion] = None,  # "Production", "Staging", "Archived" или None
        feature_names: list = None,

        additional_params: dict = None,
        log_predictions: bool = False,
        max_samples_log: int = 100,
        
        scaler = None
    ):
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            
            # ======================================================
            # =============== ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ===============
            # ======================================================
            mlflow.log_param("model_type", "Autoencoder")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # ======================================================
            # ================ МЕТРИКИ ЭКСПЕРИМЕНТА ================
            # ======================================================
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))

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
            mlflow.log_metric(threshol_name, threshold)

            # ======================================================
            # ======================= МОДЕЛЬ =======================
            # ======================================================
            # Работет тоько для моделей  [keras|sclearn]
            mlflow.keras.log_model(
                model=model.get_model_core(),
                name=model_name,
                registered_model_name=model_name,
                # signature=signature
                # input_example=X_sample[:1]  # Пример входа для Model Registry
            )

            # ======================================================
            # ============= ПЕРЕВОД В СТАДИЮ (Production) ==========
            # ======================================================
            if stage is not None:  # type: ignore
                try:
                    client = MlflowClient()
                    
                    # Получаем последнюю версию модели (только что созданную)
                    latest_versions = client.get_latest_versions(
                        model_name, 
                        stages=["None"]  # версии без стадии = только что созданные
                    )
                    
                    if latest_versions:
                        version = latest_versions[0].version
                        
                        # Если stage == "Archived", не архивируем другие
                        # Чтобы в "Production" всегда была только одна активная версия
                        archive_existing = (stage == "Production")
                        
                        client.transition_model_version_stage(
                            name=model_name,
                            version=version,
                            stage=stage,
                            archive_existing_versions=archive_existing
                        )
                        logging.info(f"Model {model_name} v{version} transferred to '{stage}'")
                    else:
                        logging.warning(f"No new version of {model_name} found for translation '{stage}'")
                        
                except Exception as e:
                    logging.error(f"Error translating model {model_name} to '{stage}': {e}")

            # ======================================================
            # ======================= SCALLER ======================
            # ======================================================
            if scaler is not None:
                try:
                    # MLflow сам упакует скалер в правильный формат
                    mlflow.sklearn.log_model(
                        scaler, 
                        artifact_path="scaler",
                        registered_model_name="scaller_save"
                        )
                    
                    logging.info("--- Scaler logged to MLflow ---")
                except Exception as e:
                    logging.info(f"--- Failed to log scaler: {e} ---")

            return run.info.run_id


#======================================================
    def _get_run_mlflow(
            self,
            model_name: str,
            experiment_name: str):
        
        client = mlflow.tracking.MlflowClient()

        

        # Поиск последнего запуска по имени эксперимента
        logging.info(f"experiment_name = {experiment_name}")
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            available = [e.name for e in client.search_experiments()]
            logging.error(
                f"Experiment '{experiment_name}' NOT FOUND! "
                f"Available: {available}"
            )
            return None

        logging.info(f"Found experiment ID: {experiment.experiment_id}")
        

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], # Тут проблема - experiment возвращает None (эксперимент ненаходится, хотя все верно указано)
            # filter_string="tags.mlflow.runName LIKE '%DenseAE%'",
            order_by=["start_time DESC"],  # Новые сверху
            max_results=1
            )

        if runs:
            run_id = runs[0].info.run_id
            logging.info(f"Latest run_id is {run_id}")
        
        # Получение данных запуска
        run = mlflow.get_run(run_id)

        return run