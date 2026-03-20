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

from src.config import (
    PATH_LOG,
    PATH_SKALERS,

    PATH_TRAIN_RAW,
    PATH_TRAIN_ADD_RAW,

    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_REPO_OWNER,
    MLFLOW_REPO_NAME
)

from pathlib import Path
from src.pipeline.pipeline import Pipeline
from src.preprocessing.scaler import Scaler
from src.preprocessing.load_data_first import LoadDataTrain
from src.preprocessing.load_data_add import LoadDataTrainAdd
from src.training.experiment import Experiment
from src.models import autoencoder
from src.training.trainer import train_model
from src.training.thresholding import choose_optimal_threshold_stadart


# ======================================================
# 1 Подготовка Loader и Scaller
# ======================================================
loader = LoadDataTrainAdd()
scaler_manager = Scaler()

pipeline = Pipeline(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir = Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    path_scaler = Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager = scaler_manager,
    loader = loader
    )

final_dataframes = pipeline.run_new()
logging.info(f"Results: final_X_train: {final_dataframes["final_X_train"]}")
logging.info(f"final_X_test: {final_dataframes["final_X_test"]}")
logging.info(f"final_X_val: {final_dataframes["final_X_val"]}")

logging.info(f"Results: final_y_train: {final_dataframes["final_y_train"]}")
logging.info(f"final_y_test: {final_dataframes["final_y_test"]}")
logging.info(f"final_y_val: {final_dataframes["final_y_val"]}")
