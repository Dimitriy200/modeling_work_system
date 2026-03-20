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
# 1 Подготовка Loader
# ======================================================
# loader = LoadDataTrain()
loader = LoadDataTrainAdd()

# ======================================================
# 2 Подготовка Scaler
# ======================================================
scaler_manager = Scaler()
scaler = scaler_manager.load_scaler(Path(PATH_SKALERS).joinpath("test_skaller.pkl"))

# ======================================================
# 3 Запуск Pipeline
# ======================================================
pipeline = Pipeline(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir = Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    path_scaler = Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager = scaler_manager,
    loader = loader
        )

# Если берем данные с датсчиков то n_anom ближе к нулю
final_train, final_test, final_valid, final_anomal = pipeline.run(n_anom=3)

logging.info(f"Results: final_train:{final_train}\n final_test:{final_test}\n final_valid:{final_valid}\n final_anomal:{final_anomal}\n")


# ======================================================
# 4 Проведение эксперимента
# ======================================================

experiment = Experiment(
    mlflow_tracking_uri = MLFLOW_TRACKING_URI,
    mlflow_repo_owner = MLFLOW_REPO_OWNER,
    mlflow_repo_name = MLFLOW_REPO_NAME,
    mlflow_username = MLFLOW_USERNAME)

MODEL_NAME = "test_model"
EXPERIMENT_NAME = "Autoencoder_Anomaly_v2"
epohs = 3
batch_size = 80

# ВАРИАНТ 1 - создаем новую модель
# encoder = autoencoder.create_default_autoencoder()

# ВАРИАНТ 2 - загружаем модел из mlflow
ld_model = experiment.load_model_from_mlflow(
    registered_model_name = MODEL_NAME
)


trained_model, history = train_model(
    # model = encoder,
    model = ld_model,
    train_df = final_train, 
    test_df = final_test, 
    epochs = epohs, 
    batch_size = batch_size)

threshold, best_accuracy, results_df = choose_optimal_threshold_stadart(
    model = trained_model,
    normal_control_df = final_valid, 
    anomaly_control_df = final_anomal)

run_id = experiment.send_experiment_to_mlflow(
    model = trained_model,
    training_history = history,

    X_train = final_train,
    X_test = final_test,
    X_val = final_valid,
    X_anomaly = final_anomal,

    threshold = threshold,
    threshold_accuracy = best_accuracy,
    df_threshold_results = results_df,

    experiment_name = EXPERIMENT_NAME,
    registered_model_name = MODEL_NAME,
    epochs = epohs,
    batch_size = batch_size)