# ======================================================
#   КЛАСС ДЛЯ ПРОЕДОБРАБОТКИ ДАННЫХ.
# 
#   ТУТ НОРМАЛИЗАЦИЯ И СТАНДАРТИЗАЦИЯ ДАННЫХ.
#   ОБУЧЕНИЕ PIPELINE
#   ДАЛЕЕ ПРОИСХОДИТ РАЗДЕЛЕНИЕ ДАННЫХ НА НОРМАЛЬНЫЕ И АНОМАЛЬНЫЕ.
#   
# ======================================================
import pandas as pd

from typing import Dict, List, Any
from sklearn.pipeline import Pipeline


class Preprocess:
    
    def __init__(self, pipeline: Pipeline  = None):
        
        if pipeline is None:
            self.pipeline = Pipeline()
        else:
            self.pipeline = pipeline
    
    # ======================================================
    def fit_pipeline(pipeline: Pipeline, dtaframe: pd.DataFrame):
        '''
        Пайплайн должен обучатья только на полном наборе данных
        '''
        return pipeline.fit(dtaframe)
    
    # ======================================================