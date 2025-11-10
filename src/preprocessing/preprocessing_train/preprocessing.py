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
import logging

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
        n_anom: int = 10
    ) -> pd.DataFrame:
         
        required_cols = ['time in cycles']
        unit_col = 'unit number'

        required_cols.append(unit_col)

        # Проверка наличия обязательных столбцов
        missing = [col for col in required_cols if col not in dtaframe.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing}")

        # Работаем с копией, чтобы не мутировать исходный df
        df_out = dtaframe.copy()

        # Сортируем по юниту и времени — критически важно!
        df_out = df_out.sort_values([unit_col, 'time in cycles']).reset_index(drop=True)

        # Помечаем последние `n_anom` записей в каждом юните
        df_out['is_anom'] = (
            df_out.groupby(unit_col)
                .cumcount(ascending=False)  # 0 — последняя запись в группе
                .lt(n_anom)                 # True для последних n_anom записей
        )

        # Логируем статистику
        total = len(df_out)
        anom_count = df_out['is_anom'].sum()
        units = df_out[unit_col].nunique()
        avg_per_unit = total / units if units > 0 else 0

        logging.info(
            f"different_norm_anom: обработано {units} юнитов, "
            f"всего {total} записей, аномалий = {anom_count} ({anom_count/total:.1%})"
        )
        
        # Опционально: проверка, что аномалии действительно в конце по времени
        # (например, для нескольких случайных юнитов)
        # if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        #     sample_units = df_out[unit_col].drop_duplicates().sample(min(3, units), random_state=42).tolist()
        #     for u in sample_units:
        #         unit_data = df_out[df_out[unit_col] == u]
        #         anom_times = unit_data[unit_data['is_anom'] == 1]['time in cycles'].values
        #         all_times = unit_data['time in cycles'].values
        #         logging.debug(f"Юнит {u}: всего {len(unit_data)} записей, аномалии на временах: {anom_times[-min(3, len(anom_times)):]} (последние)")

        return df_out

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
    
    # ======================================================
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