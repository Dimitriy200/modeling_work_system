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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocess:
    
    def __init__(self, pipeline: Pipeline  = None):
        
        if pipeline is None:
            self.pipeline = Pipeline()
        else:
            self.pipeline = pipeline
    
    # ======================================================
    def different_train_test(
              self,
              dtaframe: pd.DataFrame, 
              save_directory: str = None
            
            ) ->  tuple[pd.DataFrame, pd.DataFrame] | None:
        '''
        Разделяет данные на TRAIN и TEST выборки.
        Если указан параметри save_directory - сохраняет в формат .csv, иначе возвращает в качестве Pandas наборов.
        '''
        Train, Test = train_test_split(dtaframe)


    # ======================================================
    def fit_pipeline(
            self,
            dtaframe: pd.DataFrame, 
            pipeline: Pipeline = None
        ) -> Pipeline | None:
        '''
        Пайплайн обучается только на полном наборе данных.
        '''

        if pipeline is None:
            pipeline = Pipeline()
        
        return pipeline.fit(dtaframe)
    

    def save_pipeline(self, save_pipeline_directory: str, pipeline: Pipeline) -> None:
        '''
        Сохраняет pipeline в указанную дирректорию.
        Метод не знает o существовании указанной директории. 
        Убедитесь, что перед запуском был запущен config.py из которого можно получить путь по директории pipeline.
        '''
        with open(save_pipeline_directory, 'wb') as handle:
                    save_pik_pipeline = pickle.dumps(pipeline)
    
    # ======================================================