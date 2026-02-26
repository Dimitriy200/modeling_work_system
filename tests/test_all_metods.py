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

from src.training.trainer import train_and_log_to_mlflow
from src.models import autoencoder
from src.training.thresholding import choose_optimal_threshold
# ======================================================

import os
import dagshub
import logging


path_raw_data = Path(parent_dir).joinpath("data").joinpath("train").joinpath("raw")
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

dagshub.init(repo_owner='Dimitriy200', repo_name='modeling_work_system', mlflow=True)
encoder = autoencoder.create_default_autoencoder()

model = train_and_log_to_mlflow(
    train_df = final_train,
    test_df = final_test,
    valid_df = final_anomal,
    model = encoder,
    experiment_name = "test_all_preprocess_line",
    registered_model_name = "test_model",
    epochs = 3)

logging.info(" --- ОБУЧЕНИЕ МОДЕЛИ И СОХРАЕНИЕ ЛОГОВ В MLFLOW ЗАВЕРШЕНО --- ")

optimal_line, optimal_df = choose_optimal_threshold(
    model = model,
    normal_control_df = final_valid,
    anomaly_control_df = final_anomal,
    )

logging.info(optimal_df)
optimal_df.to_csv(Path(path_test_data).joinpath("optimal_df.csv"))

logging.info(" === ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТА ЗАВЕРНШЕНО === ")