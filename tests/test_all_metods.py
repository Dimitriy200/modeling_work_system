# ======================================================
# ТЕСТ ПОЛНОГО ЦИКЛА ДВИЖЕНИЯ ДАННЫХ
# ======================================================

import os
import dagshub
import mlflow
import logging

# ============ ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ==============
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
from modeling_work_system.preprocessing.preprocessing import Preprocess
from modeling_work_system.preprocessing.load_data_first import LoadDataTrain
from modeling_work_system.preprocessing.load_data_add import LoadDataTrainAdd
from modeling_work_system.experiment.experiment import Experiment
from modeling_work_system.models.autoencoders import autoencoder
from modeling_work_system.training.thresholding import choose_optimal_threshold_un

# ======================================================


# ======================================================
# 1. Объявляем загрузчик данных и запускаем процесс загрузки
# ======================================================
logging.info(" === НАЧАЛО ЭТАПА ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ === ")

loader = LoadDataTrain()
raw_df = loader.data_raw_load(PATH_TRAIN_RAW)

# logging.info(raw_df)
logging.info(" --- ЧТЕНИЕ ДАННЫХ ЗАВЕРШЕНО --- ")


# ======================================================
# 2. Процесс обработки данных
# ======================================================

# 2.1 Удаление пропусков
processor = Preprocess()
no_null_df = processor.delete_nan(raw_df)

# logging.info(no_null_df)
logging.info(" --- УДАЛЕНИЕ ПРОПУКОВ ЗАВЕРШЕНО --- ")


# 2.2 Определение Norm и Anom и добавление столбца с меткой
is_anom_df = processor.marking_norm_anom(no_null_df)
# logging.info(is_anom_df)
logging.info(" --- МАРКИРОВКА НОРМАЛЬНЫХ И АНОМАЛЬНЫХ ДАННЫХ ЗАВЕРШЕНА --- ")

# 2.3 Раздление Norm и Anom. Удаление столбца
# norm_df, anom_df = processor.split_norm_anom(is_anom_df)
marking_df = processor.marking_norm_anom(no_null_df)
result_dataframes = processor.split_by_engine_train_test_val(dataframe=marking_df)
logging.info(" --- MARKING OF NORMAL AND ANOMAL DATA IS COMPLETE --- ")
# logging.info(norm_df)
# logging.info(anom_df)
logging.info(" --- DATA DISTRIBUTION TO ENGINES IS COMPLETE --- ")


# ======================================================
# 2.4 Обучение и сериализация Scaler
# ======================================================

scaler_manager = Scaler()
cols = raw_df.columns.tolist()
std_scaler = scaler_manager.fit_scaler(result_dataframes["X_train"], cols) # Обучаем Scaller только на нормальных данных!!!

scaler_manager.save_scaler(Path(PATH_SKALERS).joinpath("test_skaller_v2.pkl"), std_scaler)
logging.info(" --- ОБУЧЕНИЕ И СОХРАНЕНИЕ SCALER ЗАВЕРШЕНО --- ")

# 2.5 Чтение Scaler из файла
loading_scaler = scaler_manager.load_scaler(Path(PATH_SKALERS).joinpath("test_skaller.pkl"))
# logging.info(loading_scaler)
logging.info(" --- ЧТЕНИЕ SCALER ЗАВЕРШЕНО --- ")

# 2.6 Применение scaler к NORM и ANOM
cols = norm_df.columns.tolist()
scaing_norm = scaler_manager.apply_scaler(loading_scaler, norm_df, cols)
scaing_anom = scaler_manager.apply_scaler(loading_scaler, anom_df, cols)

# logging.info(" --------- Scaling NORM --------- ")
# logging.info(scaing_norm)
# logging.info(" --------- Scaling ANOM --------- ")
# logging.info(scaing_anom)
logging.info(" --- Применение SCALER к NORM и ANOM ЗАВЕРШЕНО --- ")

# 2.7.1 Разделение на Train и Test выборки нормального набора
scaling_norm_train, scaling_process_norm_test = processor.split_train_test_standart(scaing_norm)
logging.info(" --- РАЗДЕЛЕНИЕ НА TRAIN И TEST ЗАВЕРШЕНО --- ")

# 2.7.2 Разделение Train на Normal_Train и Normal_Valid для равного набора данных с Normal_Valid = Anomal_valid
scaling_norm_test, scaling_norm_valid = processor.split_train_test_standart(scaling_process_norm_test, test_size = scaing_anom.shape[0])
logging.info(" --- РАЗДЕЛЕНИЕ TRAIN НА TRAIN И VALID ЗАВЕРШЕНО --- ")

# 2.8 Преобразование в numpy
final_train = processor.pd_to_numpy(scaling_norm_train)
final_test = processor.pd_to_numpy(scaling_norm_test)
final_valid = processor.pd_to_numpy(scaling_norm_valid)
final_anomal = processor.pd_to_numpy(scaing_anom)
# logging.info(final_train)
# logging.info(final_test)
# logging.info(anomal_valid)
logging.info(" --- ПРЕОБРАЗОВАНИЕ В NUMPY ЗАВЕРШЕНО --- ")
logging.info(" === ЭТАП ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ ЗАВЕРШЕН === ")


# ======================================================
# 3 Проведение эксперимента
# ======================================================

logging.info(" === НАЧАЛО ЭТАПА ЭКСПЕРИМЕНТОВ === ")

# 3.1 Конфигурация
experiment = Experiment(
    mlflow_tracking_uri = MLFLOW_TRACKING_URI,
    mlflow_repo_owner = MLFLOW_REPO_OWNER,
    mlflow_repo_name = MLFLOW_REPO_NAME,
    mlflow_username = MLFLOW_USERNAME,
    mlflow_pass = MLFLOW_REPO_PASSWORD,
    mlflow_token = MLFLOW_REPO_TOKEN
)

# dagshub.init(
#     repo_owner = 'Dimitriy200', 
#     repo_name = 'modeling_work_system', 
#     mlflow = True)

encoder = autoencoder.create_default_autoencoder()
epohs = 3
batch_size = 80
MODEL_NAME = "test_model"
EXPERIMENT_NAME = "Autoencoder_Anomaly_v2"

# 3.2 Обучение
trained_model, history = train_model(
    model = encoder, 
    train_df = final_train, 
    test_df = final_test, 
    epochs = epohs, 
    batch_size = batch_size)

# 3.3 Подбор порога
threshold, best_accuracy, results_df = choose_optimal_threshold_stadart(
    model = trained_model,
    normal_control_df = final_valid, 
    anomaly_control_df = final_anomal
    )

# 3.4 Сохранение логов в mlflow
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
    batch_size = batch_size
    )

logging.info(" --- ОБУЧЕНИЕ МОДЕЛИ И СОХРАЕНИЕ ЛОГОВ В MLFLOW ЗАВЕРШЕНО --- ")

logging.info(" === ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТА ЗАВЕРНШЕНО === ")


# ======================================================
# 4 Тестирование пайплайна на данных датчиков
# ======================================================

logging.info(" === НАЧАЛО ЭТАПА ДООБУЧЕНИЯ === ")
batch_size_train_add = 10

# 4.1 Выгрузить актуальную модель
loaded_model = experiment.load_model_from_mlflow(registered_model_name = MODEL_NAME)
logging.info(" --- ВЫГРУЗКА МОДЕЛИ ИЗ MLFLOW ЗАВЕРШЕНА --- ")

# Сравним модели
res = compare_weights(loaded_model, trained_model)
logging.info(f"РЕЗУЛЬТАТ СРАВНЕНИЯ ИДЕНТИЧНОСТИ ЗАГРУЖЕННОЙ И ВЫГРУЖЕННОЙ МОДЕЛЕЙ --- {res}")

# 4.2 Загрузить данные, пришедшие с датчиков
loader_add = LoadDataTrainAdd()
detector_df = loader_add.data_raw_load(Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"))
logging.info(detector_df)
logging.info(" --- ЗАГРУЗКА ДАННЫХ ИЗ ДАТЧИКОВ ЗАВЕРШЕНА --- ")

# 4.3 Предобработать данные с использованием предобученного Scaller
scaing_detector_df = scaler_manager.apply_scaler(loading_scaler, detector_df, cols)

scaing_detector_df_train, scaing_detector_df_test =  processor.split_train_test_standart(scaing_detector_df)

final_scaing_detector_df_train = processor.pd_to_numpy(scaing_detector_df_train)
final_scaing_detector_df_test = processor.pd_to_numpy(scaing_detector_df_test)
logging.info(f"final_scaing_detector_df_train\n{final_scaing_detector_df_train}")
logging.info(f"final_scaing_detector_df_test\n{final_scaing_detector_df_test}")

logging.info(" --- ПРИМЕНЕНИЕ SCALER К ДАННЫМ ИЗ ДАТЧИКОВ ЗАВЕРШЕНО --- ")

# 4.4 Дообучить модель и сохранить эксперимент
# 4.4.1 Обучение
trained_add_model, history_add = train_model(
    model = loaded_model, 
    train_df = final_scaing_detector_df_train,
    test_df = final_scaing_detector_df_test,
    epochs = epohs, 
    batch_size = batch_size_train_add)

# 4.4.2 Подбор порога
threshold_add_tarin, best_accuracy_add_tarin, results_df_add_tarin = choose_optimal_threshold_stadart(
    model = trained_add_model,
    normal_control_df = final_valid,    # valid и anomal оставляем исходные для корректного сравнения прогресса
    anomaly_control_df = final_anomal)

# 4.4.3 Сохранение логов в mlflow
run_id = experiment.send_experiment_to_mlflow(
    model = trained_add_model,
    training_history = history_add,

    X_train = final_scaing_detector_df_train,
    X_test = final_scaing_detector_df_test,
    X_val = final_valid,
    X_anomaly = final_anomal,

    threshold = threshold_add_tarin,
    threshold_accuracy = best_accuracy_add_tarin,
    df_threshold_results = results_df_add_tarin,

    experiment_name = EXPERIMENT_NAME,
    registered_model_name = MODEL_NAME,
    epochs = epohs,
    batch_size = batch_size)

logging.info(" === ЭТАП ДООБУЧЕНИЯ ЗАВЕРШЕН === ")