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
    MLFLOW_REPO_PASSWORD
)

from pathlib import Path
from modeling_work_system.pipeline.pipeline import Pipeline
from modeling_work_system.preprocessing.scaler import Scaler
from modeling_work_system.preprocessing.load_data_first import LoadDataTrain
from modeling_work_system.preprocessing.load_data_add import LoadDataTrainAdd
from modeling_work_system.experiment.experiment import Experiment
from modeling_work_system.training.thresholding import choose_optimal_threshold_un

from modeling_work_system.models.autoencoder import AutoEncoder
from modeling_work_system.models.zscoredetector import ZScoreDetector
from modeling_work_system.models.isolation_forest_detector import IsolationForestDetector


# ======================================================
# 1 Подготовка Loader и Scaller
# ======================================================
loader = LoadDataTrainAdd()
scaler_manager = Scaler()


pipeline = Pipeline(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    # path_scaler=Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager=scaler_manager,
    loader=loader
    )

# ======================================================
# 2 Предобработка данных
# ======================================================
final_dataframes = pipeline.run_new()

# ======================================================
# 3 Проведение эксперимента
# ======================================================

# СОЗДАЕМ ЭКСПЕРИМЕНТЫ ПОД КАЖДЫЙ ВИД МОДЕЛИ
# experiment_AE = Experiment(
#     mlflow_tracking_uri=MLFLOW_TRACKING_URI,
#     mlflow_repo_owner=MLFLOW_REPO_OWNER,
#     mlflow_repo_name=MLFLOW_REPO_NAME,
#     mlflow_username=MLFLOW_USERNAME,
#     mlflow_pass=MLFLOW_REPO_PASSWORD,
#     mlflow_token=MLFLOW_REPO_TOKEN,

#     train_data=final_dataframes,

#     model_name='test_ae_model',
#     experiment_name='test_ae_experiment'
# )

# experiment_z1 = Experiment(
#     mlflow_tracking_uri=MLFLOW_TRACKING_URI,
#     mlflow_repo_owner=MLFLOW_REPO_OWNER,
#     mlflow_repo_name=MLFLOW_REPO_NAME,
#     mlflow_username=MLFLOW_USERNAME,
#     mlflow_pass=MLFLOW_REPO_PASSWORD,
#     mlflow_token=MLFLOW_REPO_TOKEN,

#     train_data=final_dataframes,

#     model_name='test_z1_model',
#     experiment_name='test_z1_experiment'
# )

experiment_IF = Experiment(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN,

    train_data=final_dataframes,

    model_name='test_IF_model',
    experiment_name='test_IF_experiment'
)


# СОЗДАЕМ ВСЕ ВИДЫ МОДЕЛЕЙ
# ======================================================
# model_autoencoder = AutoEncoder()
# model_z1_score = ZScoreDetector() # Нет истории обучения
model_is = IsolationForestDetector()
# ======================================================

# ОБУЧАЕМ МОДЕЛИ
# ======================================================
# result_ae = model_autoencoder.fit(
#     X_train=final_dataframes['X_train'],
#     X_test=final_dataframes['X_val'])

# model_z1_score_fit = model_z1_score.fit(X_train=final_dataframes['X_train'])

model_isf = model_is.fit(
    X_train=final_dataframes['X_train'],
    y_train=final_dataframes['y_train'],
    X_val=final_dataframes['X_val']
)
# ======================================================

# ПОДБОР ЗНАЧЕНИЯ РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ
# ======================================================
# results_threshold_ae = choose_optimal_threshold_un(
#     model=result_ae['model'],
#     X_val=final_dataframes['X_val'],
#     y_val=final_dataframes['y_val']
# )

# results_threshold_z1 = choose_optimal_threshold_un(
#     model=model_z1_score_fit,
#     X_val=final_dataframes['X_val'],
#     y_val=final_dataframes['y_val']
# )
# ======================================================

# ЛОГИРУЕМ В MLFLOW
# ======================================================
# run_id_ae = experiment_AE.send_experiment_to_mlflow_new(
#     model=result_ae["model"],
#     training_history=result_ae["history"],
#     split_data=final_dataframes,
#     threshold_result=results_threshold_ae
# )

# run_id_z1 = experiment_z1.send_experiment_to_mlflow_new(
#     model=model_z1_score_fit,
#     training_history=None,
#     split_data=final_dataframes,
#     threshold_result=results_threshold_z1
# )

run_id_if = experiment_IF.send_experiment_to_mlflow_new(
    model=model_isf,
    training_history=None,
    split_data=final_dataframes,
    threshold_result=None
)
# ======================================================