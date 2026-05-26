import pandas as pd
import numpy as np
import logging

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from modeling_work_system.config import (
    PATH_LOG,

    RUN_ID_SCALLER,
    RUN_ID_TRESHOLD,

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
from modeling_work_system.preprocessing.load_data_add import LoadDataTrainAdd
from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.models.autoencoders.autoencoder import AutoEncoder

# ======================================================
# I Подготовка сервисов
# ======================================================
loader = LoadDataTrainAdd()
scaler_manager = Scaler()
metrics = ExperimentMetric()
# model_ae = AutoEncoder()

mlfs = Mlflowservice(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN
    )

# ======================================================
# II Скачивание модели, разделяющей поверхности и scaller-a
# ======================================================
model_core = mlfs.load_model_from_mlflow()
threshold = mlfs.load_threshold_from_mlflow(run_id=RUN_ID_TRESHOLD)
scaller = mlfs.load_skaller_from_mlflow()

model_ae = AutoEncoder(model_core=model_core, threshold=threshold)


# ======================================================
# III Предобработка данных
# ======================================================
pipeline = PipelineFit(
    path_data_dir=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    scaler_manager=scaler_manager,
    loader=loader,
    scaler=scaller
    )

final_dataframes = pipeline.run()

# ======================================================
# IV Проведение эксперимента [До-обучение]
# ======================================================
train_result = model_ae.fit(
    X_train=final_dataframes["X_train"],
    X_test=final_dataframes["X_test"],
    X_val=final_dataframes["X_val"],
    Y_val=final_dataframes["y_val"]
    )

ae_metrics = metrics.compute_all_metrics(
    y_true=final_dataframes["y_test"],
    y_pred=model_ae.predict(X=final_dataframes["X_test"],threshold=train_result["threshold"]),
    scores=model_ae.predict_scores(final_dataframes["X_test"]),
    threshold=train_result["threshold"]
    )

# ======================================================
# V Логирование эксперимента в Mlflow
# ======================================================
run_id = mlfs.save_model_to_mlflow(
    model=model_ae,
    metrics=ae_metrics.to_dict(),

    training_history=train_result["history"],
    threshold=train_result["threshold"],
    epochs=train_result["threshold"],
    batch_size=train_result["batch_size"]
    )