# ======================================================
#   КЛАСС ДЛЯ ПРОЕДОБРАБОТКИ ДАННЫХ.
# 
#   ТУТ НОРМАЛИЗАЦИЯ И СТАНДАРТИЗАЦИЯ ДАННЫХ.
#   ОБУЧЕНИЕ PIPELINE
#   ДАЛЕЕ ПРОИСХОДИТ РАЗДЕЛЕНИЕ ДАННЫХ НА НОРМАЛЬНЫЕ И АНОМАЛЬНЫЕ.
#   
# ======================================================
import pandas as pd
import os
import pickle

from typing import Dict, List, Any, Tuple
# from sklearn.scaller import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocess:
    
    def __init__(self, scaller: StandardScaler  = None):
        
        if scaller is None:
            self.scaller = StandardScaler()
        else:
            self.scaller = scaller
    
    # ======================================================
    def different_train_test(
              self,
              dtaframe: pd.DataFrame, 
              save_directory: str = None,
              file_name_train: str = "train.csv",
              file_name_test: str = "test.csv"
            ) ->  tuple[pd.DataFrame, pd.DataFrame] | None:
        '''
        Разделяет данные на TRAIN и TEST выборки.
        Если указан параметри save_directory - сохраняет в формат .csv, иначе возвращает в качестве Pandas наборов.
        '''
        train, test = train_test_split(dtaframe)
        train_pd = pd.DataFrame(data=train, columns=dtaframe.columns)
        test_pd = pd.DataFrame(data=test, columns=dtaframe.columns)
        
        if save_directory is None:    
            return train_pd, test_pd
        else:
            train_pd.to_csv(path_or_buf = os.path.join(save_directory, file_name_train))
            test_pd.to_csv(path_or_buf = os.path.join(save_directory, file_name_test))
            return None

    # ======================================================
    def different_norm_anom(
        self,
        dtaframe: pd.DataFrame,

    )->

    # ======================================================
    def fit_scaller(
            self,
            dtaframe: pd.DataFrame, 
            scaller: StandardScaler = None
        ) -> StandardScaler | None:
        '''
        Пайплайн обучается только на полном наборе данных.
        '''

        if scaller is None:
            scaller = StandardScaler()
        
        return scaller.fit(dtaframe)
    

    def save_scaller(
            self, 
            save_pipeline_directory: str, 
            scaller: StandardScaler
        ) -> None:
        '''
        Сохраняет scaller в указанную дирректорию.
        Метод не знает o существовании указанной директории. 
        Убедитесь, что перед запуском был запущен config.py из которого можно получить путь по директории scaller.
        '''
        with open(save_pipeline_directory, 'wb') as handle:
                    save_pik_pipeline = pickle.dumps(scaller)
    
    # ======================================================