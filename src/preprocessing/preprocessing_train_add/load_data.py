# ======================================================
#   КЛАСС ОБРАБОДКИ ДАННЫХ ДЛЯ ДООБУЧЕНИЯ
# 
# ======================================================

import pandas as pd
import numpy as np
import os
import pickle
import logging
import json

from pathlib import Path
from sklearn.base import BaseEstimator
from typing import Dict, List, Any, Tuple, Optional, Type


class LoadDataTrainAdd:

    def data_raw_load(
            self,
            directory_input_path: str,
            directory_out_path: str = None
        )-> pd.DataFrame | None:
        '''
        Считывает JSON-файлы из иерархической директории:
        <directory_input_path>/<unit_id>/<file>.json
        Каждый JSON-файл должен быть словарём числовых значений.
        Последние 3 значения в каждом файле отбрасываются (как в исходном коде).
        '''

        directory = Path(directory_input_path)
    
        if not directory.exists():
            logging.error(f"Директория не найдена: {directory}")
            return None

        res_rows = []
        
        try:
            # Сортируем unit_dirs по имени (лексикографически)
            unit_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
            logging.info(f"Найдено {len(unit_dirs)} unit-директорий в {directory}")

            if not unit_dirs:
                logging.warning("Unit-директории не найдены.")
                return pd.DataFrame()

            for unit_dir in unit_dirs:
                unit_id = unit_dir.name
                json_files = sorted([f for f in unit_dir.iterdir() if f.suffix.lower() == '.json'])
                
                if not json_files:
                    logging.debug(f"B unit '{unit_id}' нет JSON-файлов.")
                    continue

                logging.info(f"Unit '{unit_id}': {len(json_files)} JSON-файлов")

                for json_path in json_files:
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        if not isinstance(data, dict):
                            logging.warning(f"Файл {json_path} не содержит словарь — пропуск.")
                            continue

                        values = list(data.values())
                        if len(values) < 3:
                            logging.warning(f"Файл {json_path} имеет <3 значений — пропуск.")
                            continue

                        # Отбрасываем последние 3 значения, как в оригинале
                        values = values[:-3]
                        keys = list(data.keys())

                        # Преобразуем в float (с обработкой ошибок)
                        try:
                            numeric_values = [float(v) for v in values]
                        except (ValueError, TypeError) as e:
                            logging.error(f"Ошибка преобразования в float в {json_path}: {e}")
                            continue

                        # Добавляем служебные поля
                        row = {
                            # **{f'feat_{i}': val for i, val in enumerate(numeric_values)},
                            # 'unit_id': unit_id,
                            **dict(zip(keys, numeric_values)), #  'key1': 1, key1': 1
                            # 'source_file': json_path.name
                        }
                        res_rows.append(row)

                    except json.JSONDecodeError as e:
                        logging.error(f"Некорректный JSON в {json_path}: {e}")
                    except Exception as e:
                        logging.error(f"Ошибка обработки {json_path}: {e}")

            if not res_rows:
                logging.warning("Ни один валидный JSON не был обработан.")
                return pd.DataFrame()

            jsons_combinet_df = pd.DataFrame(res_rows)
            logging.info(f"Создан DataFrame: {jsons_combinet_df.shape} (строк, столбцов). Пример столбцов: {list(jsons_combinet_df.columns[:5])}...")

            if directory_out_path is None:
                return jsons_combinet_df
            else:
                file_name_out = os.path.join(directory_out_path, 'jsons_combinet_df.csv')
                jsons_combinet_df.to_csv(path_or_buf = file_name_out)

        except Exception as e:
            logging.exception(f"Критическая ошибка в read_json_to_dataframe: {e}")
            return None
