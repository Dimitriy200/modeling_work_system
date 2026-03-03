# ======================================================
# Модуль с готовыми pipeline для проведени экспериментов
# ======================================================

import pandas as pd
import numpy as np
import pathlib
import logging

from typing import Type
from sklearn.base import BaseEstimator
from ..preprocessing.load_data_first import LoadDataTrain
from ..preprocessing.preprocessing import Preprocess
from ..preprocessing.load_data_add import LoadDataTrainAdd
from ..preprocessing.load_data import LoadData
from ..preprocessing.scaler import Scaler


class Pipeline:
    
    def __init__(
            self,

            path_data_dir: str,
            path_scaler: str,
            scaler_manager: Type[Scaler],
            loader: Type[LoadData],
            processor: Preprocess = Preprocess()
        ):

        self.scaler_manager = scaler_manager
        self.loader = loader
        self.processor = processor
        self.scaler = scaler_manager.load_scaler(path_scaler)
        
        if path_data_dir is None:
            logging.error("data_raw_dir is None!!!")
        else:
            self.data_raw_dir = path_data_dir


    def run(self):
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

        # 1 Удаление пропусков
        logging.info(" === НАЧАЛО ЭТАПА ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ === ")
        raw_df = self.loader.data_raw_load(self.data_raw_dir)
        logging.info(" --- ЧТЕНИЕ ДАННЫХ ЗАВЕРШЕНО --- ")

        # 2 Определение Norm и Anom и добавление столбца с меткой
        no_null_df = self.processor.delete_nan(raw_df)
        logging.info(" --- УДАЛЕНИЕ ПРОПУКОВ ЗАВЕРШЕНО --- ")

        # 3 Раздление Norm и Anom. Удаление столбца
        is_anom_df = self.processor.marking_norm_anom(no_null_df)
        logging.info(" --- МАРКИРОВКА НОРМАЛЬНЫХ И АНОМАЛЬНЫХ ДАННЫХ ЗАВЕРШЕНА --- ")

        norm_df, anom_df = self.processor.different_norm_anom(is_anom_df)
        logging.info(" --- РАЗДЕЛЕНИЕ НА NORM И ANOM ЗАВЕРШЕНО --- ")

        # 4 Применение Scaler к NORM и ANOM
        cols = norm_df.columns.tolist()
        scaing_norm = self.scaler_manager.use_scaler(self.scaler, norm_df, cols)
        scaing_anom = self.scaler_manager.use_scaler(self.scaler, anom_df, cols)
        logging.info(" --- Применение SCALER к NORM и ANOM ЗАВЕРШЕНО --- ")

        # 5 Разделение на Train и Test выборки нормального набора
        scaling_norm_train, scaling_process_norm_test = self.processor.different_train_test(scaing_norm)
        logging.info(" --- РАЗДЕЛЕНИЕ НА TRAIN И TEST ЗАВЕРШЕНО --- ")

        # 6 Разделение Train на Normal_Train и Normal_Valid для равного набора данных с Normal_Valid = Anomal_valid
        scaling_norm_test, scaling_norm_valid = self.processor.different_train_test(
            scaling_process_norm_test, 
            test_size = scaing_anom.shape[0]
            )
        logging.info(" --- РАЗДЕЛЕНИЕ TRAIN НА TRAIN И VALID ЗАВЕРШЕНО --- ")

        # 7 Преобразование в numpy
        final_train = self.processor.pd_to_numpy(scaling_norm_train)
        final_test = self.processor.pd_to_numpy(scaling_norm_test)
        final_valid = self.processor.pd_to_numpy(scaling_norm_valid)
        final_anomal = self.processor.pd_to_numpy(scaing_anom)
        logging.info(" --- ПРЕОБРАЗОВАНИЕ В NUMPY ЗАВЕРШЕНО --- ")
        logging.info(" === ЭТАП ПРЕДОБРАБОТКИ БОЛЬШИХ ДАННЫХ ЗАВЕРШЕН === ")

        return final_train, final_test, final_valid, final_anomal