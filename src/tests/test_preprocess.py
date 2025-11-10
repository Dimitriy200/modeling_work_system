import pandas as pd
import sys

# =============== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ===============
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from preprocessing.preprocessing_train.preprocessing import Preprocess
# ======================================================

# =============== ИМПОРТ КОНФИГА ===============
from config import PATH_TRAIN_RAW, PATH_TRAIN_PROCESSED


df_in = pd.read_csv("D:\\yniver\\modeling_work_system\\data\\tarin\\raw\\train_FD001.csv")
pr = Preprocess()

df_out = pr.different_norm_anom(dtaframe = df_in)
df_out.to_csv("D:\\yniver\\modeling_work_system\\data\\tarin\\processing\\testing_dif_anom.csv")