# ======================================================
# Модуль с готовыми pipeline для проведени экспериментов
# ======================================================

import pandas as pd
import numpy as np
import pathlib
import logging

from typing import Type, Dict
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from ..preprocessing.load_data_first import LoadDataTrain
from ..preprocessing.preprocessing import Preprocess
from ..preprocessing.load_data_add import LoadDataTrainAdd
from ..preprocessing.load_data import LoadData
from ..preprocessing.scaler import Scaler


class Pipeline:
    
    def __init__(
            self,

            path_data_dir: str,
            scaler_manager: Type[Scaler],
            loader: Type[LoadData],
            processor: Preprocess = Preprocess(),
            path_scaler: str = None

        ):

        self.scaler_manager = scaler_manager
        self.loader = loader
        self.processor = processor

        if path_scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler_manager.load_scaler(path_scaler)
        
        if path_data_dir is None:
            logging.error("data_raw_dir is None!!!")
        else:
            self.data_raw_dir = path_data_dir
    
# ======================================================
    def run_new(self, n_anom = 10) -> Dict[str, pd.DataFrame]:
        """
        Запускает процесс предобработки данных
         1. Читает данные из указанной директории. Способ чтения зависит от Loader-а.
         2. Удаляет пропуски.
         3. Маркирует и диференцирует данные на нормальные и аномальные
         4. Применяет к данным указанный Scaler.
         5. Разделяет датафрейм на TRAIN и TEST.
         6. Преобразовывает в numpy наборы данных и возвращает в порядке:
          - TRAIN
          - TEST
          - VALID
          - ANOMAL
        """

        # ======================================================
        # 1. Объявляем загрузчик данных и запускаем процесс загрузки
        # ======================================================
        logging.info(" === BEGINNING OF THE BIG DATA PREPROCESSING STAGE === ")

        raw_df = self.loader.data_raw_load(self.data_raw_dir)
        cols = raw_df.columns.tolist()

        logging.info(f"cols = {cols}")
        logging.info(" --- DATA READING COMPLETED --- ")

        # ======================================================
        # 2. Процесс обработки данных
        # ======================================================
        # 2.1 Удаление пропусков
        no_null_df = self.processor.delete_nan(raw_df)

        # logging.info(no_null_df)
        logging.info(" --- DATA DELETION COMPLETE --- ")

        # 2.2 Определение Norm и Anom и добавление столбца с меткой
        marking_df = self.processor.marking_norm_anom(no_null_df)
        # marking_df.to_csv(Path(PATH_TRAIN_PROCESSED).joinpath("marking_df.csv"))
        # logging.info(f"marking_df\n{marking_df}")
        logging.info(" --- MARKING OF NORMAL AND ANOMAL DATA IS COMPLETE --- ")

        result_dataframes = self.processor.split_by_engine_train_test_val(dataframe=marking_df)
        logging.info(" --- DATA DISTRIBUTION TO ENGINES IS COMPLETE --- ")


        # ======================================================
        # 3 Обучение и нормализация c Scaler
        # ======================================================
        std_scaler = self.scaler_manager.fit_scaler(result_dataframes['X_train'], cols) # Обучаем Scaller только на нормальных данных!!!

        final_X_train = self.scaler_manager.apply_scaler(std_scaler, result_dataframes['X_train'], cols)
        final_X__val = self.scaler_manager.apply_scaler(std_scaler, result_dataframes['X_val'], cols)
        final_X__test = self.scaler_manager.apply_scaler(std_scaler, result_dataframes['X_test'], cols)


        # ======================================================
        # 4 Финальные преобразования меток
        # ======================================================
        final_Y_train = self.processor.pd_to_numpy(result_dataframes['y_train'])
        final_Y__val = self.processor.pd_to_numpy(result_dataframes['y_val'])
        final_Y__test = self.processor.pd_to_numpy(result_dataframes['y_test'])
        

        result_dataframes.update({
            'X_train': final_X_train,
            'X_val': final_X__val,
            'X_test': final_X__test,

            'y_train': final_Y_train,
            'y_val': final_Y__val,
            'y_test': final_Y__test
            })
        
        logging.info(f"Results: final_X_train: {result_dataframes['X_train']}")
        logging.info(f"final_X_test: {result_dataframes['X_test']}")
        logging.info(f"final_X_val: {result_dataframes['X_val']}")
        logging.info(f"Results: final_y_train: {result_dataframes['y_train']}")
        logging.info(f"final_y_test: {result_dataframes['y_test']}")
        logging.info(f"final_y_val: {result_dataframes['y_val']}")

        logging.info(" --- APPLICATION OF SCALER TO TRAIN TEST AND VAL COMPLETED --- ")
        logging.info(" === BIG DATA PREPROCESSING STAGE COMPLETED === ")

        return result_dataframes