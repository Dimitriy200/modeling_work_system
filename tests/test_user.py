# ======================================================
# Тест pipeline
# ======================================================

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
    MLFLOW_REPO_PASSWORD,

    RUN_ID_TRESHOLD
)

from pathlib import Path
from modeling_work_system.pipeline.pipeline_fit import PipelineFit
from modeling_work_system.pipeline.pipeline_predict import PipelinePredict
from modeling_work_system.preprocessing.scaler import Scaler
from modeling_work_system.preprocessing.load_data_first import LoadDataTrain
from modeling_work_system.preprocessing.load_data_add import LoadDataTrainAdd
from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.models.autoencoders.autoencoder import AutoEncoder


mlfs = Mlflowservice(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN
)


# ======================================================
# 1 Подготовка Loader и Scaller
# ======================================================
loader = LoadDataTrainAdd()
scaler_manager = Scaler()
scaler = mlfs.load_skaller_from_mlflow()

pipeline_predict = PipelinePredict(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    # path_scaler=Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager=scaler_manager,
    scaler=scaler,
    loader=loader
    )

pipeline_fit = PipelineFit(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    scaler_manager=scaler_manager,
    scaler=scaler,
    loader=loader
    )

# ======================================================
# 2 Предобработка данных для дообучения
# ======================================================
final_dataframes_fit = pipeline_fit.run()

# ======================================================
# 2 Предобработка данных инференса
# ======================================================
final_dataframes_predict = pipeline_predict.run()


# ======================================================
# 3 Проведение эксперимента
# ======================================================
# ======================================================
# 3.2 До-обучение
# ======================================================
mlfs = Mlflowservice(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN
)

model_core = mlfs.load_model_from_mlflow()
model_core = mlfs.load_model_from_mlflow(stage="Production")
model_ae = AutoEncoder(model_core=model_core)
train_result = model_ae.fit(
    X_train=final_dataframes_fit["X_train"],
    X_test=final_dataframes_fit["X_test"],
    X_val=final_dataframes_fit["X_val"],
    Y_val=final_dataframes_fit["y_val"])

metrics = ExperimentMetric()
ae_metrics = metrics.compute_all_metrics(
    y_true=final_dataframes_fit["y_test"],
    y_pred=model_ae.predict(X=final_dataframes_fit["X_test"],threshold=train_result["threshold"]),
    scores=model_ae.predict_scores(final_dataframes_fit["X_test"]),
    threshold=train_result["threshold"]
    )

# Логируем эксперимент в mlflow
run_id = mlfs.save_model_to_mlflow(
    model=model_ae,
    stage="Production",
    metrics=ae_metrics.to_dict(),

    training_history=train_result["history"],
    threshold=train_result["threshold"],
    epochs=train_result["threshold"],
    batch_size=train_result["batch_size"]
    )


# ======================================================
# 3.3 Инференс
# ======================================================
mlfs = Mlflowservice(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN
)

model_core = mlfs.load_model_from_mlflow(stage="Production")
threshold = mlfs.load_threshold_from_mlflow(run_id=RUN_ID_TRESHOLD)
model_ae = AutoEncoder(model_core=model_core, threshold=threshold)

metrics = ExperimentMetric()

# Предсказание класса [НОРМА/АНОМАЛИЯ]
predicted_clases = model_ae.predict(X=final_dataframes_predict["final_X"], threshold=threshold)
