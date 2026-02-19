import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler

# =============== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ===============
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from preprocessing.preprocessing_train.preprocessing import Preprocess
# ======================================================

# =============== ИМПОРТ КОНФИГА ===============
from config import PATH_TRAIN_RAW, PATH_TRAIN_PROCESSED


# ======================================================
#   ТЕСТЫ НА БОЛЬШОМ НАБОРЕ

# df_in = pd.read_csv("D:\\yniver\\modeling_work_system\\data\\tarin\\processing\\combined_df.csv")
df_in = pd.read_csv("D:\\yniver\\modeling_work_system\\data\\train\\processing\\combined_df.csv")
print(df_in)

pr = Preprocess()

df_non_nan = pr.delete_nan(df_in)

df_out = pr.different_norm_anom(dtaframe = df_non_nan)

sensor_cols = [col for col in df_out.columns if col.startswith('sensor')]

scaler = pr.fit_scaler_on_normal(
    dataframe = df_out,
    feature_columns = sensor_cols,
    scaler_class = StandardScaler
)

df_scaled = pr.use_scaler(
    dataframe = df_out,
    scaler = scaler, 
    feature_columns = sensor_cols)

print(df_scaled.shape)

df_scaled.to_csv("D:\\yniver\\modeling_work_system\\data\\train\\final\\final.csv", index=False)
# ======================================================