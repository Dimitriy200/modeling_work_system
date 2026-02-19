import pandas as pd
import numpy as np

# =============== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ===============
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from training.trainer import train_and_log_to_mlflow
from models.autoencoder import create_default_autoencoder
# ======================================================

final_df_dir = "D:\\yniver\\modeling_work_system\\data\\train\\final\\final.csv"



mofel = train_and_log_to_mlflow(
    train_path=final_df_dir
)