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
    def split_data_by_engine(
            self,
            dataframe: pd.DataFrame,

            unit_col = 'unit_number', 
            label_col = 'is_anom',

            train_ratio = 0.6, 
            val_ratio = 0.2, 
            test_ratio = 0.2, 
            random_state = 42
        ):
        """
        Разделяет датасет на Train/Val/Test по идентификаторам двигателей (unit_number),
        а не по строкам. Это предотвращает утечку данных (data leakage).
        
        Логика разделения:
        1. Двигатели случайно распределяются между наборами.
        2. Train: Только нормальные данные (для обучения Autoencoder).
        3. Val/Test: Все данные (Норма + Аномалия) для оценки качества детекции.
        
        Parameters
        ----------
        unit_col : str
            Название колонки с ID двигателя (по умолчанию 'unit_number').
        label_col : str
            Название колонки с метками (по умолчанию 'label').
        train_ratio : float
            Доля двигателей для обучения.
        val_ratio : float
            Доля двигателей для валидации (подбор порога).
        test_ratio : float
            Доля двигателей для финального теста.
        random_state : int
            Фиксация случайности для воспроизводимости результатов.
            
        Returns
        -------
        dict
            Словарь с датасетами:
            {
                'X_train': pd.DataFrame (только норма),
                'X_val': pd.DataFrame (норма + аномалия),
                'y_val': pd.Series (метки для валидации),
                'X_test': pd.DataFrame (норма + аномалия),
                'y_test': pd.Series (метки для теста),
                'info': dict (статистика разделения)
            }
        """
        
        # 1. Проверка суммарной пропорции
        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"Сумма ratio должна быть равна 1.0. Сейчас: {total_ratio}")
            
        # 2. Получаем уникальный список двигателей и перемешиваем их
        unique_units = dataframe[unit_col].unique()
        
        # Сначала отделяем Train от остальных (Val + Test)
        train_units, temp_units = train_test_split(
            unique_units, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_state
        )
        
        # Затем делим оставшиеся на Val и Test пропорционально
        test_ratio_adjusted = test_ratio / (val_ratio + test_ratio)
        val_units, test_units = train_test_split(
            temp_units, 
            test_size=test_ratio_adjusted, 
            random_state=random_state
        )
        
        # 3. Формируем итоговые датасеты
        # TRAIN: Только нормальные данные выбранных двигателей
        mask_train = (dataframe[unit_col].isin(train_units)) & (dataframe[label_col] == 'Norm')
        df_train = dataframe.loc[mask_train].copy()
        
        # VAL: Все данные выбранных двигателей (для подбора порога)
        mask_val = dataframe[unit_col].isin(val_units)
        df_val = dataframe.loc[mask_val].copy()
        
        # TEST: Все данные выбранных двигателей (для финальной оценки)
        mask_test = dataframe[unit_col].isin(test_units)
        df_test = dataframe.loc[mask_test].copy()
        
        # 4. Sanity Checks (Проверка на утечку данных)
        # Проверка пересечения двигателей
        assert len(set(train_units) & set(val_units)) == 0, "Утечка: двигатели Train и Val пересекаются!"
        assert len(set(train_units) & set(test_units)) == 0, "Утечка: двигатели Train и Test пересекаются!"
        assert len(set(val_units) & set(test_units)) == 0, "Утечка: двигатели Val и Test пересекаются!"
        
        # Проверка чистоты обучения (в Train не должно быть аномалий)
        unique_labels_train = df_train[label_col].unique()
        assert 'Anom' not in unique_labels_train, "Ошибка: В обучающей выборке обнаружены аномалии!"
        
        # 5. Подготовка словаря для возврата
        # X_... содержат все признаки (включая метку, если нужно, или только фичи - зависит от вашей архитектуры)
        # y_... содержат только метки для удобства расчета метрик
        result = {
            'X_train': df_train.drop(columns=[label_col]),
            'y_train': df_train[label_col],
            'X_val': df_val.drop(columns=[label_col]),
            'y_val': df_val[label_col],
            'X_test': df_test.drop(columns=[label_col]),
            'y_test': df_test[label_col],
            'info': {
                'n_train_units': len(train_units),
                'n_val_units': len(val_units),
                'n_test_units': len(test_units),
                'n_train_samples': len(df_train),
                'n_val_samples': len(df_val),
                'n_test_samples': len(df_test),
                'train_units': train_units,
                'val_units': val_units,
                'test_units': test_units
            }
        }
        
        # 6. Вывод статистики в консоль
        print("="*50)
        print("DATA SPLIT STATISTICS (BY ENGINE ID)")
        print("="*50)
        print(f"Train: {len(train_units)} engines, {len(df_train)} samples (Norm only)")
        print(f"Val:   {len(val_units)} engines, {len(df_val)} samples (Norm + Anom)")
        print(f"Test:  {len(test_units)} engines, {len(df_test)} samples (Norm + Anom)")
        print("="*50)
        
        # Сохраняем информацию в атрибут класса для доступа из других методов
        self.split_info = result['info']
        
        return result

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
