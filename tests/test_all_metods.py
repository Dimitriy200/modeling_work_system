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
# ======================================================

import os
import dagshub

path_raw_data = Path(parent_dir).joinpath("data").joinpath("train").joinpath("raw")
path_scaler = Path(parent_dir).joinpath("skalers").joinpath("test_sca;er.pkl")


# 1. Объявляем загрузчик данных и запускаем процесс загрузки
print(" === НАЧАЛО ЭТАПА ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ === ")

loader = LoadDataTrain()
raw_df = loader.data_raw_load(path_raw_data)

# print(raw_df)
print(" --- ЧТЕНИЕ ДАННЫХ ЗАВЕРШЕНО --- ")


# 2. Процесс обработки данных
# 2.1 Удаление пропусков
preprocessor = Preprocess()
no_null_df = preprocessor.delete_nan(raw_df)

# print(no_null_df)
print(" --- УДАЛЕНИЕ ПРОПУКОВ ЗАВЕРШЕНО --- ")


# 2.2 Определение Norm и Anom и добавление столбца с меткой
is_anom_df = preprocessor.marking_norm_anom(no_null_df)
# print(is_anom_df)
print(" --- МАРКИРОВКА НОРМАЛЬНЫХ И АНОМАЛЬНЫХ ДАННЫХ ЗАВЕРШЕНА --- ")

# 2.3 Раздление Norm и Anom. Удаление столбца
norm_df, anom_df = preprocessor.different_norm_anom(is_anom_df)
# print(norm_df)
# print(anom_df)
print(" --- РАЗДЕЛЕНИЕ НА NORM И ANOM ЗАВЕРШЕНО --- ")

# 2.4 Обучение и сериализация Scaler
cols = norm_df.columns
scaler = preprocessor.fit_scaler(norm_df, cols)
preprocessor.save_scaler(path_scaler, scaler)
print(" --- ОБУЧЕНИЕ И СОХРАНЕНИЕ SCALER ЗАВЕРШЕНО --- ")

# 2.5 Чтение Scaler из файла
loading_scaler = preprocessor.load_scaler(path_scaler)
# print(loading_scaler)
print(" --- ЧТЕНИЕ SCALER ЗАВЕРШЕНО --- ")

# 2.6 Применение scaler к NORM и ANOM
cols = norm_df.columns.tolist()
scaing_norm = preprocessor.use_scaler(loading_scaler, norm_df, cols)
scaing_anom = preprocessor.use_scaler(loading_scaler, anom_df, cols)

# print(" --------- Scaling NORM --------- ")
# print(scaing_norm)
# print(" --------- Scaling ANOM --------- ")
# print(scaing_anom)
print(" --- Применение SCALER к NORM и ANOM ЗАВЕРШЕНО --- ")

# 2.7 Разделение на Train и Test выборки нормального набора
scaling_norm_train, scaling_norm_test = preprocessor.different_train_test(scaing_norm)
print(" --- РАЗДЕЛЕНИЕ НА NORM TRAIN И TEST ЗАВЕРШЕНО --- ")

# 2.8 Преобразование в numpy
final_train, final_test, anomal_valid = preprocessor.pd_to_numpy(scaling_norm_train, scaling_norm_test, scaing_anom)
# print(final_train)
# print(final_test)
# print(anomal_valid)
print(" --- ПРЕОБРАЗОВАНИЕ В NUMPY ЗАВЕРШЕНО --- ")
print(" === ЭТАП ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ ЗАВЕРШЕН === ")


# 3 Проведение эксперимента
print(" === НАЧАЛО ЭТАПА ЭКСПЕРИМЕНТОВ === ")

dagshub.init(repo_owner='Dimitriy200', repo_name='modeling_work_system', mlflow=True)
encoder = autoencoder.create_default_autoencoder()

model = train_and_log_to_mlflow(
    train_df = final_train,
    test_df = final_test,
    valid_df = anomal_valid,
    model = encoder,
    experiment_name = "test_all_preprocess_line",
    registered_model_name = "test_model",
    epochs = 3)

print(" --- ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТА ЗАВЕРНШЕНО --- ")