# ======================================================
# ТЕСТ ПОЛНОГО ЦИКЛА ДВИЖЕНИЯ ДАННЫХ
# ======================================================


# ============ ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ==============
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from src.preprocessing.preprocessing_train.load_data import LoadDataTrain
from src.preprocessing.preprocessing_train.preprocessing import Preprocess
from src.preprocessing.preprocessing_train_add.load_data import LoadDataTrainAdd

from src.training.trainer import train_model, compare_weights
from src.models import autoencoder
from src.training.thresholding import choose_optimal_threshold
from src.training.mlflow_loader import log_run_to_mlflow, load_model_from_mlflow
# ======================================================

import os
import dagshub
import mlflow
import logging


path_raw_data = Path(parent_dir).joinpath("data").joinpath("train").joinpath("raw")
path_raw_data_detectors = Path(parent_dir).joinpath("data").joinpath("train_add").joinpath("raw").joinpath("2024-07-02_2024-07-03_2024-07-04")

path_scaler = Path(parent_dir).joinpath("skalers").joinpath("test_sca;er.pkl")
path_test_data = Path(parent_dir).joinpath("data").joinpath("test_data")
path_logs = Path(parent_dir).joinpath("logs")

logging.basicConfig(
    level = logging.INFO,
    filename = Path(path_logs).joinpath('tests_logs.log'),
    filemode = "w",
    format = "%(asctime)s %(levelname)s %(message)s"
)


# ======================================================
# 1. Объявляем загрузчик данных и запускаем процесс загрузки
# ======================================================
logging.info(" === НАЧАЛО ЭТАПА ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ === ")

loader = LoadDataTrain()
raw_df = loader.data_raw_load(path_raw_data)

# logging.info(raw_df)
logging.info(" --- ЧТЕНИЕ ДАННЫХ ЗАВЕРШЕНО --- ")


# ======================================================
# 2. Процесс обработки данных
# ======================================================

# 2.1 Удаление пропусков
preprocessor = Preprocess()
no_null_df = preprocessor.delete_nan(raw_df)

# logging.info(no_null_df)
logging.info(" --- УДАЛЕНИЕ ПРОПУКОВ ЗАВЕРШЕНО --- ")


# 2.2 Определение Norm и Anom и добавление столбца с меткой
is_anom_df = preprocessor.marking_norm_anom(no_null_df)
# logging.info(is_anom_df)
logging.info(" --- МАРКИРОВКА НОРМАЛЬНЫХ И АНОМАЛЬНЫХ ДАННЫХ ЗАВЕРШЕНА --- ")

# 2.3 Раздление Norm и Anom. Удаление столбца
norm_df, anom_df = preprocessor.different_norm_anom(is_anom_df)
# logging.info(norm_df)
# logging.info(anom_df)
logging.info(" --- РАЗДЕЛЕНИЕ НА NORM И ANOM ЗАВЕРШЕНО --- ")

# 2.4 Обучение и сериализация Scaler
cols = norm_df.columns
scaler = preprocessor.fit_scaler(norm_df, cols)
preprocessor.save_scaler(path_scaler, scaler)
logging.info(" --- ОБУЧЕНИЕ И СОХРАНЕНИЕ SCALER ЗАВЕРШЕНО --- ")

# 2.5 Чтение Scaler из файла
loading_scaler = preprocessor.load_scaler(path_scaler)
# logging.info(loading_scaler)
logging.info(" --- ЧТЕНИЕ SCALER ЗАВЕРШЕНО --- ")

# 2.6 Применение scaler к NORM и ANOM
cols = norm_df.columns.tolist()
scaing_norm = preprocessor.use_scaler(loading_scaler, norm_df, cols)
scaing_anom = preprocessor.use_scaler(loading_scaler, anom_df, cols)

# logging.info(" --------- Scaling NORM --------- ")
# logging.info(scaing_norm)
# logging.info(" --------- Scaling ANOM --------- ")
# logging.info(scaing_anom)
logging.info(" --- Применение SCALER к NORM и ANOM ЗАВЕРШЕНО --- ")

# 2.7.1 Разделение на Train и Test выборки нормального набора
scaling_norm_train, scaling_process_norm_test = preprocessor.different_train_test(scaing_norm)
logging.info(" --- РАЗДЕЛЕНИЕ НА TRAIN И TEST ЗАВЕРШЕНО --- ")

# 2.7.2 Разделение Train на Normal_Train и Normal_Valid для равного набора данных с Normal_Valid = Anomal_valid
scaling_norm_test, scaling_norm_valid = preprocessor.different_train_test(scaling_process_norm_test, test_size = scaing_anom.shape[0])
logging.info(" --- РАЗДЕЛЕНИЕ TRAIN НА TRAIN И VALID ЗАВЕРШЕНО --- ")

# 2.8 Преобразование в numpy
final_train = preprocessor.pd_to_numpy(scaling_norm_train)
final_test = preprocessor.pd_to_numpy(scaling_norm_test)
final_valid = preprocessor.pd_to_numpy(scaling_norm_valid)
final_anomal = preprocessor.pd_to_numpy(scaing_anom)
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
dagshub.init(
    repo_owner = 'Dimitriy200', 
    repo_name = 'modeling_work_system', 
    mlflow = True)

encoder = autoencoder.create_default_autoencoder()
epohs = 3
batch_size = 80
registered_model_name = "test_model"
experiment_name = "Autoencoder_Anomaly_v2"

# 3.2 Обучение
trained_model = train_model(
    model = encoder, 
    train_df = final_train, 
    test_df = final_test, 
    epochs = epohs, 
    batch_size = batch_size)

# 3.3 Подбор порога
threshold, best_accuracy, results_df = choose_optimal_threshold(
    model = trained_model, 
    normal_control_df = final_valid, 
    anomaly_control_df = final_anomal)

# 3.4 Сохранение логов в mlflow
run_id = log_run_to_mlflow(
    model = trained_model,
    X_train = final_train,
    X_test = final_test,
    X_val = final_valid,
    X_anomaly = final_anomal,

    threshold = threshold,
    threshold_accuracy = best_accuracy,
    df_threshold_results = results_df,

    experiment_name = experiment_name,
    registered_model_name = registered_model_name,
    epochs = epohs,
    batch_size = batch_size)

logging.info(" --- ОБУЧЕНИЕ МОДЕЛИ И СОХРАЕНИЕ ЛОГОВ В MLFLOW ЗАВЕРШЕНО --- ")

logging.info(" === ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТА ЗАВЕРНШЕНО === ")


# ======================================================
# 4 Тестирование пайплайна на данных датчиков
# ======================================================

logging.info(" === НАЧАЛО ЭТАПА ДООБУЧЕНИЯ === ")
batch_size_train_add = 10


# 4.1 Выгрузить актуальную модель
loaded_model = load_model_from_mlflow(registered_model_name = registered_model_name)
logging.info(" --- ВЫГРУЗКА МОДЕЛИ ИЗ MLFLOW ЗАВЕРШЕНА --- ")

# Сравним модели
res = compare_weights(loaded_model, trained_model)
logging.info(f"РЕЗУЛЬТАТ СРАВНЕНИЯ ИДЕНТИЧНОСТИ ЗАГРУЖЕННОЙ И ВЫГРУЖЕННОЙ МОДЕЛЕЙ --- {res}")

# 4.2 Загрузить данные, пришедшие с датчиков
loader_add = LoadDataTrainAdd()
detector_df = loader_add.data_raw_load(path_raw_data_detectors)
logging.info(detector_df)
logging.info(" --- ЗАГРУЗКА ДАННЫХ ИЗ ДАТЧИКОВ ЗАВЕРШЕНА --- ")

# 4.3 Предобработать данные с использованием предобученного Scaller
scaing_detector_df = preprocessor.use_scaler(loading_scaler, detector_df, cols)

scaing_detector_df_train, scaing_detector_df_test =  preprocessor.different_train_test(scaing_detector_df)

final_scaing_detector_df_train = preprocessor.pd_to_numpy(scaing_detector_df_train)
final_scaing_detector_df_test = preprocessor.pd_to_numpy(scaing_detector_df_test)
logging.info(f"final_scaing_detector_df_train\n{final_scaing_detector_df_train}")
logging.info(f"final_scaing_detector_df_test\n{final_scaing_detector_df_test}")

logging.info(" --- ПРИМЕНЕНИЕ SCALER К ДАННЫМ ИЗ ДАТЧИКОВ ЗАВЕРШЕНО --- ")

# 4.4 Дообучить модель и сохранить эксперимент
# 4.4.1 Обучение
trained_add_model = train_model(
    model = loaded_model, 
    train_df = final_scaing_detector_df_train,
    test_df = final_scaing_detector_df_test,
    epochs = epohs, 
    batch_size = batch_size_train_add)

# 4.4.2 Подбор порога
threshold_add_tarin, best_accuracy_add_tarin, results_df_add_tarin = choose_optimal_threshold(
    model = trained_add_model,
    normal_control_df = final_valid,    # valid и anomal оставляем исходные для корректного сравнения прогресса
    anomaly_control_df = final_anomal)

# 4.4.3 Сохранение логов в mlflow
run_id = log_run_to_mlflow(
    model = trained_add_model,
    X_train = final_scaing_detector_df_train,
    X_test = final_scaing_detector_df_test,
    X_val = final_valid,
    X_anomaly = final_anomal,

    threshold = threshold_add_tarin,
    threshold_accuracy = best_accuracy_add_tarin,
    df_threshold_results = results_df_add_tarin,

    experiment_name = experiment_name,
    registered_model_name = registered_model_name,
    epochs = epohs,
    batch_size = batch_size)

logging.info(" === ЭТАП ДООБУЧЕНИЯ ЗАВЕРШЕН === ")