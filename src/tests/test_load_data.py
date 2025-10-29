import pandas as pd
import sys

# ===== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ =====
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from preprocessing.preprocessing_train.load_data import LoadDataTrain

from config import PATH_TRAIN_RAW


ld = LoadDataTrain()
df = ld.data_raw_load(PATH_TRAIN_RAW)
df.to_csv("test_load_data.csv")