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

from typing import Dict, List, Any, Tuple, Optional, Type, TypedDict
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
        dataframe: pd.DataFrame,
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
        missing = [col for col in required_cols if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing}")

        # Работаем с копией, чтобы не мутировать исходный dataframe
        dataframe_out = dataframe.copy()

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
    def split_norm_anom(
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
    def split_train_test_standart(
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
    def split_by_engine_train_test_val(
            self,
            dataframe: pd.DataFrame, 
            unit_col: str = 'unit number', 
            label_col: str = 'is_anom', 
            train_ratio: float = 0.6, 
            val_ratio: float = 0.2, 
            test_ratio: float = 0.2, 
            random_state: int = 42
            ) -> Dict[str, pd.DataFrame]:
        """
        Разделяет датасет на Train/Test/Val по идентификаторам двигателей.
        Возвращает данные в формате pandas DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            Исходный размеченный датасет.
        unit_col : str
            Название колонки с ID двигателя.
        label_col : str
            Название колонки с метками ('Norm'/'Anom').
        train_ratio, val_ratio, test_ratio : float
            Пропорции разделения двигателей.
        random_state : int
            Фиксация случайности.
            
        Returns
        -------
        dict : {
            'X_train': pd.DataFrame, 'y_train': pd.Series,
            'X_val': pd.DataFrame, 'y_val': pd.Series,
            'X_test': pd.DataFrame, 'y_test': pd.Series,
            'info': dict
        }
        """
    
        # Проверка колонок
        if unit_col not in dataframe.columns:
            raise ValueError(f"Колонка '{unit_col}' не найдена! Доступны: {dataframe.columns.tolist()}")
        if label_col not in dataframe.columns:
            raise ValueError(f"Колонка '{label_col}' не найдена! Доступны: {dataframe.columns.tolist()}")
            
        # Определение меток
        # value_counts = data[label_col].value_counts
        if label_col in dataframe.columns:
            normal_label = False
            anomaly_label = True
            logging.info(f"[INFO] Обнаружена колонка '{label_col}'. Используем булеву логику: False=Norm, True=Anom.")
            
        # Разделение двигателей
        unique_units = dataframe[unit_col].unique()
        
        if len(unique_units) < 3:
            raise ValueError(f"Слишком мало двигателей ({len(unique_units)}) для разделения!")
        
        train_units, temp_units = train_test_split(
            unique_units, 
            test_size = (val_ratio + test_ratio), 
            random_state = random_state
        )
        
        test_ratio_adjusted = test_ratio / (val_ratio + test_ratio)
        
        val_units, test_units = train_test_split(
            temp_units, 
            test_size = test_ratio_adjusted, 
            random_state = random_state
        )
        
        # Фильтрация данных
        mask_train = (dataframe[unit_col].isin(train_units)) & (dataframe[label_col] == normal_label)
        df_train = dataframe.loc[mask_train].copy()
        
        df_val = dataframe.loc[dataframe[unit_col].isin(val_units)].copy()
        df_test = dataframe.loc[dataframe[unit_col].isin(test_units)].copy()
        
        # Проверка на пустой Train
        if df_train.empty:
            raise ValueError(
                f"Train набор пуст! Проверьте: метка нормы='{normal_label}', "
                f"колонка метки='{label_col}', данные размечены."
            )
        
        # Формирование результата
        result_dataframes = {
            'X_train': df_train.drop(columns=[label_col]).reset_index(drop=True),
            'y_train': df_train[label_col].reset_index(drop=True),
            'X_val': df_val.drop(columns=[label_col]).reset_index(drop=True),
            'y_val': df_val[label_col].reset_index(drop=True),
            'X_test': df_test.drop(columns=[label_col]).reset_index(drop=True),
            'y_test': df_test[label_col].reset_index(drop=True)
        }
        
        result_info = {
            # 'normal_label': normal_label,
            # 'anomaly_label': anomaly_label,
            'n_train_units': len(train_units),
            'n_val_units': len(val_units),
            'n_test_units': len(test_units),
            'n_train_samples': len(df_train),
            'n_val_samples': len(df_val),
            'n_test_samples': len(df_test),
            'train_units': list(train_units),
            'val_units': list(val_units),
            'test_units': list(test_units)
        }
        
        # Логирование 
        logging.info("=== RESULTS OF DATA SEPARATION BY ENGINES ===")
        logging.info(f"count train units = {result_info['n_train_units']}")
        logging.info(f"count val units = {result_info['n_val_units']}")
        logging.info(f"count test units = {result_info['n_test_units']}")
        logging.info(f"count train samples = {result_info['n_train_samples']}")
        logging.info(f"count val samples = {result_info['n_val_samples']}")
        logging.info(f"count test samples = {result_info['n_test_samples']}")
        logging.info(f"X_train is pd.DataFrame = {isinstance(result_dataframes['X_train'], pd.DataFrame)}")
        logging.info(f"y_train is pd.DataFrame = {isinstance(result_dataframes['y_train'], pd.DataFrame)}")
        logging.info(f"X_val is pd.DataFrame = {isinstance(result_dataframes['X_val'], pd.DataFrame)}")
        logging.info(f"y_val is pd.DataFrame = {isinstance(result_dataframes['y_val'], pd.DataFrame)}")
        logging.info(f"X_test is pd.DataFrame = {isinstance(result_dataframes['X_test'], pd.DataFrame)}")
        logging.info(f"y_test is pd.DataFrame = {isinstance(result_dataframes['y_test'], pd.DataFrame)}")
        
        return result_dataframes
        
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
