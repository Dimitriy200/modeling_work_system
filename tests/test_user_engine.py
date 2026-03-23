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
    MLFLOW_REPO_NAME,
    MLFLOW_REPO_TOKEN,
    MLFLOW_REPO_PASSWORD
)

from pathlib import Path
from src.pipeline.pipeline import Pipeline
from src.preprocessing.scaler import Scaler
from src.preprocessing.load_data_first import LoadDataTrain
from src.preprocessing.load_data_add import LoadDataTrainAdd
# from src.training.experiment import Experiment
from src.training.experiment_new import Experiment
from src.models import autoencoder
from src.training.trainer import train_model
from src.training.thresholding import choose_optimal_threshold_stadart, choose_optimal_threshold_un


# ======================================================
# 1 Подготовка Loader и Scaller
# ======================================================
loader = LoadDataTrainAdd()
scaler_manager = Scaler()

pipeline = Pipeline(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    path_scaler=Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager=scaler_manager,
    loader=loader
    )

# ======================================================
# 2 Предобработка данных
# ======================================================
final_dataframes = pipeline.run_new()
logging.info(f"Results: final_X_train: {final_dataframes["X_train"]}")
logging.info(f"final_X_test: {final_dataframes["X_test"]}")
logging.info(f"final_X_val: {final_dataframes["X_val"]}")

logging.info(f"Results: final_y_train: {final_dataframes["y_train"]}")
logging.info(f"final_y_test: {final_dataframes["y_test"]}")
logging.info(f"final_y_val: {final_dataframes["y_val"]}")

# ======================================================
# 3 Проведение эксперимента
# ======================================================

experiment = Experiment(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN,
    train_data=final_dataframes
)

ld_model = experiment.load_model_from_mlflow()

trained_model, history = experiment.train_model(
    # model = encoder,
    model=ld_model,
    train_df=final_dataframes["X_train"], 
    test_df=final_dataframes["X_val"]
)

results_threshold = choose_optimal_threshold_un(
    model=trained_model,
    X_val=final_dataframes["X_val"],
    y_val=final_dataframes["y_val"]
)

run_id = experiment.send_experiment_to_mlflow_new(
    model=trained_model,
    training_history=history,
    split_data=final_dataframes,
    threshold_result=results_threshold
)