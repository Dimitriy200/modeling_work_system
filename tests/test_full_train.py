import pandas as pd
import numpy as np
import logging

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from modeling_work_system.config import (
    PATH_LOG,
    PATH_SKALERS,

    PATH_TRAIN_RAW,
    PATH_TRAIN_ADD_RAW,

    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_REPO_OWNER,
    MLFLOW_REPO_NAME,
    MLFLOW_REPO_TOKEN,
    MLFLOW_REPO_PASSWORD
)

from pathlib import Path
from modeling_work_system.pipeline.pipeline_fit import PipelineFit
from modeling_work_system.preprocessing.scaler import Scaler
from modeling_work_system.preprocessing.load_data_first import LoadDataTrain
from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult
from modeling_work_system.models.autoencoders.autoencoder import AutoEncoder


# ======================================================
# I Подготовка сервисов
# ======================================================
loader = LoadDataTrain()
scaler_manager = Scaler()
metrics = ExperimentMetric()
model_ae = AutoEncoder()

pipeline = PipelineFit(
    path_data_dir = PATH_TRAIN_RAW,
    scaler_manager=scaler_manager,
    loader=loader
    )

mlfs = Mlflowservice(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN
)


# ======================================================
# II Предобработка данных
# ======================================================
final_pipeline = pipeline.run(fit_scaller=True)


# ======================================================
# III Проведение эксперимента
# ======================================================

# Обучение
train_result = model_ae.fit(
    X_train=final_pipeline["X_train"],
    X_test=final_pipeline["X_test"],
    X_val=final_pipeline["X_val"],
    Y_val=final_pipeline["y_val"])

# Сбор метрик
ae_metrics = metrics.compute_all_metrics(
    y_true=final_pipeline["y_test"],
    y_pred=model_ae.predict(X=final_pipeline["X_test"],threshold=train_result["threshold"]),
    scores=model_ae.predict_scores(final_pipeline["X_test"]),
    threshold=train_result["threshold"]
    )

# ======================================================
# IV Логирование данных
# ======================================================
run_id = mlfs.save_model_to_mlflow(
    model=model_ae,
    metrics=ae_metrics.to_dict(),

    training_history=train_result["history"],
    threshold=train_result["threshold"],
    epochs=train_result["threshold"],
    batch_size=train_result["batch_size"],
    scaler=final_pipeline["scaller"],

    experiment_name="scaller_save",
    model_name="scaller_save"
    )