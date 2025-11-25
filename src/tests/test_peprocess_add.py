import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler

# =============== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ===============
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from preprocessing.preprocessing_train_add.load_data import LoadDataTrainAdd
# ======================================================

# =============== ИМПОРТ КОНФИГА ===============
from config import PATH_TRAIN_RAW, PATH_TRAIN_PROCESSED

jsons_dir = "D:\\yniver\\modeling_work_system\\data\\train_add\\raw\\2024-07-02_2024-07-03_2024-07-04"
ld = LoadDataTrainAdd()
ld.data_raw_load(directory_out_path=jsons_dir)