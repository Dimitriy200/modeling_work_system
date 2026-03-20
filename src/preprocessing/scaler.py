# ======================================================
# КЛАСС ДЛЯ ОРГАНИЗАЦИИ РАБОТЫ SCALLER
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

class Scaler():
    def __init__(self):
        pass

    # ======================================================
    def fit_scaler(
            self,
            dataframe: pd.DataFrame,
            feature_columns: List[str],
            scaler_class: Type[BaseEstimator] = StandardScaler,
            scaler_kwargs: Optional[dict] = None
        ) -> Type[BaseEstimator] | None:
        '''
        Обучает scaler на наборе НОРМАЛЬНЫХ данных.
        Ha других наборах обучение исклчено.
        dataframe обязан содержать столбец is_anom.
        '''
        # # Проверка наличия 'is_anom'
        # if 'is_anom' not in dataframe.columns:
        #     raise ValueError("Столбец 'is_anom' отсутствует. Сначала вызовите different_norm_anom().")
        
        # missing_cols = [col for col in feature_columns if col not in dataframe.columns]
        # if missing_cols:
        #     raise ValueError(f"Отсутствующие столбцы: {missing_cols}")
        
        # non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
        # if non_numeric:
        #     raise ValueError(f"Нечисловые столбцы: {non_numeric}. Scaler требует числовые признаки.")
        
        # df_normal = dataframe[dataframe['is_anom'] == False]
        # n_norm, n_total = len(df_normal), len(dataframe)
        # if n_norm == 0:
        #     raise ValueError("Нет нормальных данных (is_anom == False). Проверьте разметку.")
        
        # logging.info(f"Обучение scaler на {n_norm} нормальных записях ({n_norm / n_total:.1%} от общего)")
        
        scaler_kwargs = scaler_kwargs or {}

        try:
            scaler = scaler_class(**scaler_kwargs)
            scaler.fit(dataframe[feature_columns])          # scaler.fit(df_normal[feature_columns])
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении scaler'a {scaler_class.__name__}: {e}") from e

        logging.info(f"Scaler {scaler_class.__name__} has been trained on the normal data. Features: {feature_columns}")

        return scaler
    
    # ======================================================
    def save_scaler(
            self, 
            save_scaler_directory: str, 
            scaler: Type[BaseEstimator]
        ) -> None:
        '''
        Сохраняет scaler в указанную дирректорию.
        Метод не знает o существовании указанной директории. 
        Убедитесь, что перед запуском был запущен config.py из которого можно получить путь по директории scaler.
        Разрешение сохраняемого файла должно быть .pkl
        '''
        with open(save_scaler_directory, 'wb') as handle:
                    pickle.dump(scaler, handle)
    
    # ======================================================
    def load_scaler(self,
            scaler_directory: str ) -> Type[BaseEstimator]:

        scaler_path = Path(scaler_directory)
        
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler не найден по пути: {scaler_path.resolve()}"
            )

        try:
            with open(scaler_path, 'rb') as file_scaller:
                scaler = pickle.load(file_scaller)
                logging.info(f"Scaler успешно загружен из: {scaler_path}")
        
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке scaler'a: {e}") from e
        
        # Валидация: должен быть совместим со sklearn API
        if not hasattr(scaler, 'transform'):
            raise TypeError(
                f"Загруженный объект не поддерживает .transform(). Тип: {type(scaler)}"
            )
        if not hasattr(scaler, 'fit'):  # опционально, но полезно
            logging.warning("Загруженный scaler не имеет .fit() — дообучение невозможно.")
        
        # Логируем тип и параметры (если есть)
        logging.info(f"Тип scaler'a\a: {scaler.__class__.__name__}")
        if hasattr(scaler, 'n_features_in_'):
            logging.info(f"Ожидаемое число признаков: {scaler.n_features_in_}")
        
        return scaler

    # ======================================================
    def apply_scaler(
            self,
            scaler: Type[BaseEstimator],
            dataframe: pd.DataFrame,
            feature_columns: Optional[List[str]] = None
        ) -> pd.DataFrame:
        '''
        Метод использует указанный scaler 
        для нормализации и стандартизации входного dataframe.
        '''
        
        if not hasattr(scaler, 'transform'):
            raise ValueError("Переданный scaler не имеет метода .transform()")
        
        # Столбцы, которые НЕ должны нормализоваться (служебные / категориальные / метки)
        # exclude_cols = {
        #     'unit number'
        # }

            # Определяем признаки для нормализации
        if feature_columns is None:
            # Берём все числовые столбцы, кроме исключённых
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
            # feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            logging.debug(f"Автоматически выбраны числовые признаки для нормализации: {feature_columns}")
        else:
            # Проверяем наличие
            missing = [col for col in feature_columns if col not in dataframe.columns]
            if missing:
                raise ValueError(f"Следующие столбцы отсутствуют в датафрейме: {missing}")
            # Проверяем числовость
            non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
            if non_numeric:
                raise ValueError(f"Нечисловые столбцы в feature_columns: {non_numeric}. "
                                "Scaler работает только с числовыми данными.")

        # Проверяем, что есть что нормализовать
        if not feature_columns:
            logging.warning("Нет столбцов для нормализации. Возвращается копия исходного датафрейма.")
            return dataframe.copy()

        logging.info(f"Применение scaler к {len(feature_columns)} столбцам: {feature_columns}")

        # Работаем с копией
        dataframe_out = dataframe.copy()

        try:
            # Извлекаем данные для трансформации
            X = dataframe_out[feature_columns].values  # shape: (n_samples, n_features)

            # Применяем scaler
            X_scaled = scaler.transform(X)

            # Записываем обратно — сохраняя исходные имена столбцов и индекс
            dataframe_out[feature_columns] = pd.DataFrame(
                X_scaled,
                columns=feature_columns,
                index=dataframe_out.index
            )

                # Проверка: убедимся, что dtype стал float (если scaler даёт float)
            for col in feature_columns:
                if not pd.api.types.is_float_dtype(dataframe_out[col]):
                    dataframe_out[col] = dataframe_out[col].astype(np.float32)  # или float64, если точность критична

            logging.info(f"Scaler успешно применён. Пример значений для {feature_columns[0]}: "
                        f"{dataframe_out[feature_columns[0]].iloc[:3].tolist()}")
            
        except Exception as e:
            logging.error(f"Ошибка при применении scaler: {e}")
            raise

        return dataframe_out

    # ======================================================