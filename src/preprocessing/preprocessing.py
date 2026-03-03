# ======================================================
#   КЛАСС ДЛЯ ПРОЕДОБРАБОТКИ ДАННЫХ.
# 
#   РЕАЛИЗУЕТ МЕТОДЫ НОРМАЛИЗАЦИИ И СТАНДАРТИЗАЦИИ ДАННЫХ,
#   РАЗДЕЛЕНИЯ ДАННЫХ НА НОРМАЛЬНЫЕ И АНОМАЛЬНЫЕ.
# ======================================================
import pandas as pd
import numpy as np
import os
import pickle
import logging


from typing import Dict, List, Any, Tuple, Optional, Type
# from sklearn.scaler import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from pathlib import Path


class Preprocess:
    
    def __init__(self):
        pass
    
    # ======================================================
    def delete_nan(
            self,
            dataframe: pd.DataFrame) -> pd.DataFrame:
        
        # Удаляем строки с None
        initial_rows = len(dataframe)
        dataframe.dropna(inplace=True)
        print(f"Удалено строк с None: {initial_rows - len(dataframe)}")

        # Финальная проверка
        print(f"Размер dataframe: {dataframe.shape}")
        print(f"Остались ли NAN: {dataframe.isna().any().any()}")

        return dataframe

    # ======================================================
    def marking_norm_anom(
        self,
        dtaframe: pd.DataFrame,
        n_anom: int = 10
    ) -> pd.DataFrame:
        
        '''
        Добавляет столбец is_anom со значениями аномальных и нормальных циклов = True и False соответственно.
        По умолчанию - последние 10 циклов каждого двигатея считаются аномальными.
        Разделение данных предлагается вынести за пределы функции в соображениях сохранения безопасности метода.
        '''
         
        required_cols = ['time in cycles']
        unit_col = 'unit number'

        required_cols.append(unit_col)

        # Проверка наличия обязательных столбцов
        missing = [col for col in required_cols if col not in dtaframe.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing}")

        # Работаем с копией, чтобы не мутировать исходный dataframe
        dataframe_out = dtaframe.copy()

        # Сортируем по юниту и времени — критически важно!
        dataframe_out = dataframe_out.sort_values([unit_col, 'time in cycles']).reset_index(drop=True)

        # Помечаем последние `n_anom` записей в каждом юните
        dataframe_out['is_anom'] = (
            dataframe_out.groupby(unit_col)
                .cumcount(ascending=False)  # 0 — последняя запись в группе
                .lt(n_anom)                 # True для последних n_anom записей
        )

        # Логируем статистику
        total = len(dataframe_out)
        anom_count = dataframe_out['is_anom'].sum()
        units = dataframe_out[unit_col].nunique()
        avg_per_unit = total / units if units > 0 else 0

        logging.info(
            f"different_norm_anom: обработано {units} юнитов, "
            f"всего {total} записей, аномалий = {anom_count} ({anom_count/total:.1%})"
        )
        
        # Опционально: проверка, что аномалии действительно в конце по времени
        # (например, для нескольких случайных юнитов)
        # if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        #     sample_units = dataframe_out[unit_col].drop_duplicates().sample(min(3, units), random_state=42).tolist()
        #     for u in sample_units:
        #         unit_data = dataframe_out[dataframe_out[unit_col] == u]
        #         anom_times = unit_data[unit_data['is_anom'] == 1]['time in cycles'].values
        #         all_times = unit_data['time in cycles'].values
        #         logging.debug(f"Юнит {u}: всего {len(unit_data)} записей, аномалии на временах: {anom_times[-min(3, len(anom_times)):]} (последние)")

        return dataframe_out

    # ======================================================
    def different_norm_anom(
            self,
            dataframe: pd.DataFrame
        ):

        '''
        Датасет должен содержать столбец "is_anom" 
        Метод разделяет единый набор данных на поднаборы с нормальными и аномальными данными.
        Столбец "is_anom" удаляется.
        '''

        # Проверяем, есть ли в наборе столбец is_anom

        if dataframe.columns.isin(['is_anom']).any():
            normal_data = dataframe[dataframe['is_anom'] == False].copy()
            anomal_data = dataframe[dataframe['is_anom'] == True].copy()

            # 4. Удаляем целевую колонку
            normal_data = normal_data.drop(columns = ['is_anom'])
            anomal_data = anomal_data.drop(columns = ['is_anom'])

            return normal_data, anomal_data
        else:
            logging.info("Отсутствует столбец is_anom")
            print("Отсутствует столбец is_anom")
            
            return 0

    # ======================================================
    def different_train_test(
              self,
              dtaframe: pd.DataFrame,
              test_size: float | None = None,
              train_size: float | None = None,
              save_directory: str = None,
              file_name_train: str = "train.csv",
              file_name_test: str = "test.csv"
            ) ->  tuple[pd.DataFrame, pd.DataFrame] | None:
        '''
        Разделяет данные на TRAIN и TEST выборки.
        Если указан параметри save_directory - сохраняет в формат .csv, иначе возвращает в качестве Pandas наборов.
        '''
        train, test = train_test_split(
            dtaframe,
            test_size=test_size,
            train_size=train_size
        )
        train_pd = pd.DataFrame(data=train, columns=dtaframe.columns)
        test_pd = pd.DataFrame(data=test, columns=dtaframe.columns)
        
        if save_directory is None:
            return train_pd, test_pd
        else:
            train_pd.to_csv(path_or_buf = os.path.join(save_directory, file_name_train), index=False)
            test_pd.to_csv(path_or_buf = os.path.join(save_directory, file_name_test), index=False)
            return None

    # ======================================================
    def pd_to_numpy(
            self,
            dataframe :pd.DataFrame ):
        
        if not dataframe.empty:
            return dataframe.to_numpy()
        else:
            logging.info("dataframe пуст")
            print("dataframe пуст")

            return None
