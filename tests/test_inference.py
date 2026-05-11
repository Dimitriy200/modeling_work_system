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
from modeling_work_system.pipeline.pipeline import Pipeline
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
data = loader.data_raw_load(
    directory_input_path=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04")
    )

scaler_manager = Scaler()

processed_data = scaler_manager.apply_scaler(
    scaler=scaller,
    dataframe=data
)


# ======================================================
# IV Проведение эксперимента [До-обучение]
# ======================================================
predict_result = model_ae.predict(X=processed_data, threshold=threshold)
logging.info(f"predict_result = {predict_result}")