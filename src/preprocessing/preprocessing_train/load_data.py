import os
import pickle
import pandas as pd
import json
import csv
import sys
import logging

from typing import Dict, List, Any
# Импорт модуля config. 
# Данный модуль находится выше на две директории - отсюда и заморочки.
from pathlib import Path
# parent_dir = Path(__file__).parent.parent.parent
# sys.path.append(str(parent_dir))
# import config


class LoadDataTrain:

    # =============================================================================
    def read_csv_generator(self, directory_path: str):
        '''
        Генератор для чтения файлов из директории один за другим
        '''

        directory = Path(directory_path)
        
        for csv_file in directory.glob("*.csv"):
            try:
                df = pd.read_csv(
                    csv_file,
                    dtype=float
                )
                yield df, csv_file.name
            except Exception as e:
                logging.error(f"Ошибка чтения файла {csv_file}: {e}")
                continue
    
    # =============================================================================
    def data_raw_load(self, directory_path: str) -> pd.DataFrame:
        
        csv_files = list(Path(directory_path).glob("*.csv"))
        logging.info(f"CSV files found: {len(csv_files)}")

        if not csv_files:
            logging.warning("CSV files not found")
            return pd.DataFrame()

        # Генератор для чтения файлов
        data_frames = []
        for df, filename in self.read_csv_generator(directory_path):
            logging.info(f"Writed csv file {filename}: {df.shape}")
            df['source_file'] = filename
            data_frames.append(df)
        
        combined_df = pd.concat(data_frames, ignore_index=False)
        logging.info(f"Combined DF paams: {combined_df.shape}")

        return combined_df
    # =============================================================================
