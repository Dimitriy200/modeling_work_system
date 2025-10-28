import sys
# Импорт модуля config. 
# Данный модуль находится выше на две директории - отсюда и заморочки.
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
from config import pd
from config import PATH_TRAIN_RAW
from config import main_logger


class LoadDataTrain:
    # =============================================================================
    def read_csv_generator(directory_path):
        '''
        Генератор для чтения файлов из директории один за другим 
        [экономим память, ленивая загрузка, и тд]
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
                main_logger.error(f"Ошибка чтения файла {csv_file}: {e}")
                continue
    # =============================================================================

    # =============================================================================
    def data_raw_load(directory_path: str = PATH_TRAIN_RAW) -> pd.DataFrame:
        
        csv_files = list(Path(directory_path).glob("*.csv"))
        main_logger.info(f"CSV files found: {len(csv_files)}")

        if not csv_files:
            main_logger.warning("CSV files not found")
            return pd.DataFrame()

        # Генератор для чтения файлов
        data_frames = []
        for df, filename in read_csv_generator(directory_path):
            main_logger.info(f"Writed csv file {filename}: {df.shape}")
            df['source_file'] = filename
            data_frames.append(df)
        
        combined_df = pd.concat(data_frames, ignore_index=False)
        main_logger.info(f"Combined DF paams: {combined_df.shape}")

        return combined_df
    # =============================================================================