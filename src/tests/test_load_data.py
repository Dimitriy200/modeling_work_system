import pandas as pd
import sys

from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from preprocessing.preprocessing_train import load_data as ld


df = ld.data_raw_load(ld.PATH_TRAIN_RAW)
df.to_csv("test_load_data.csv")