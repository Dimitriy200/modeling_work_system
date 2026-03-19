# ======================================================
# Тест split_data_by_engine
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
    PATH_TRAIN_PROCESSED,
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
from src.training.experiment import Experiment
from src.preprocessing.preprocessing import Preprocess

from src.models import autoencoder
from src.training.trainer import train_model
from src.training.thresholding import choose_optimal_threshold_stadart, choose_optimal_threshold_un


# ======================================================
# 1. Объявляем загрузчик данных и запускаем процесс загрузки
# ======================================================
logging.info(" === BEGINNING OF THE BIG DATA PREPROCESSING STAGE === ")

loader = LoadDataTrain()
raw_df = loader.data_raw_load(PATH_TRAIN_RAW)
cols = raw_df.columns

# logging.info(raw_df)
logging.info(" --- DATA READING COMPLETED --- ")

# ======================================================
# 2. Процесс обработки данных
# ======================================================

# 2.1 Удаление пропусков
preprocessor = Preprocess()
no_null_df = preprocessor.delete_nan(raw_df)

# logging.info(no_null_df)
logging.info(" --- GAP REMOVAL COMPLETED --- ")

# 2.2 Определение Norm и Anom и добавление столбца с меткой
marking_df = preprocessor.marking_norm_anom(no_null_df)
# marking_df.to_csv(Path(PATH_TRAIN_PROCESSED).joinpath("marking_df.csv"))
# logging.info(f"marking_df\n{marking_df}")
logging.info(" --- MARKING OF NORMAL AND ABNORMAL DATA IS COMPLETE --- ")

X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocessor.divide_by_engine_train_test_val(
    dataframe = marking_df)

logging.info()
logging.info(" --- DATA DISTRIBUTION TO ENGINES IS COMPLETE --- ")

# df_train, df_test, df_val = preprocessor.split_by_engine(split_data)
# np.savetxt(Path(PATH_TRAIN_PROCESSED).joinpath("df_train.csv"), df_train, delimiter=',')

# logging.info(" --- ENGINE DATA SPLITTING COMPLETE --- ")


# ======================================================
# 2.4 Обучение и нормализация c Scaler
# ======================================================

scaler_manager = Scaler()
# Обучаем Scaller только на нормальных данных!!!
std_scaler = scaler_manager.fit_scaler(X_test, cols)

scaling_train = scaler_manager.use_scaler(std_scaler, X_train, cols)
scaling_val = scaler_manager.use_scaler(std_scaler, X_val, cols)
scaling_test = scaler_manager.use_scaler(std_scaler, X_test, cols)

logging.info(" --- APPLICATION OF SCALER TO TRAIN TEST AND VAL COMPLETED --- ")


# ======================================================
# 3 Проведение эксперимента
# ======================================================

# logging.info(" === BEGINNING OF THE EXPERIMENTAL STAGE === ")

# experiment = Experiment(
#     mlflow_tracking_uri = MLFLOW_TRACKING_URI,
#     mlflow_repo_owner = MLFLOW_REPO_OWNER,
#     mlflow_repo_name = MLFLOW_REPO_NAME,
#     mlflow_username = MLFLOW_USERNAME)

# MODEL_NAME = "test_model"
# EXPERIMENT_NAME = "Autoencoder_Anomaly_v2"
# epohs = 3
# batch_size = 80

# # ВАРИАНТ 2 - загркжаем модел из mlflow
# ld_model = experiment.load_model_from_mlflow(
#     registered_model_name = MODEL_NAME
# )

# # 3.1 Конфигурация
# experiment = Experiment(
#     mlflow_tracking_uri = MLFLOW_TRACKING_URI,
#     mlflow_repo_owner = MLFLOW_REPO_OWNER,
#     mlflow_repo_name = MLFLOW_REPO_NAME,
#     mlflow_username = MLFLOW_USERNAME,
#     mlflow_pass = MLFLOW_REPO_PASSWORD,
#     mlflow_token = MLFLOW_REPO_TOKEN
# )

# trained_model, history = train_model(
#     model = ld_model,
#     train_df = scaling_train, 
#     test_df = scaling_test, 
#     epochs = epohs, 
#     batch_size = batch_size
# )

# threshold_result = choose_optimal_threshold_un(
#     model = trained_model,
#     X_val = split_data['X_val'],      # DataFrame с признаками
#     y_val = split_data['y_val'],      # Series с метками
#     feature_names = cols,  # только сенсоры
#     metric='f1',                    # или 'recall', если важнее не пропускать поломки
#     target_recall=0.95,             # хотим найти 95% реальных аномалий
#     plot=True,                      # построить графики для статьи
#     run_id='exp_001'
# )

# run_id = experiment.send_experiment_to_mlflow_new(
#     model = trained_model,
#     training_history=history,
#     split_data = split_data,
#     threshold_result = threshold_result,
#     experiment_name = "Turbofan_AnomalyDetection",
#     registered_model_name = "LSTM_Autoencoder_CMAPSS",
#     epochs = 50,
#     batch_size = 256,
#     feature_names = cols,
#     additional_params = {
#         "anomaly_window": 10,
#         "model_architecture": "LSTM_AE",
#         "optimizer": "adam",
#         "loss": "mse"
#     },
#     log_predictions=True
# )