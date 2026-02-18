import pandas as pd
import numpy as np

# =============== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ===============
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from training.trainer import train_and_log_to_mlflow
# ======================================================

