# ======================================================
# Тест pipeline
# ======================================================

import pandas as pd
import numpy as np
import logging

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from src.config import (
    PATH_SKALERS,
    PATH_TRAIN_RAW,
    PATH_LOG,

    )
from pathlib import Path
from src.pipeline.pipeline import Pipeline
from src.preprocessing.scaler import Scaler
from src.preprocessing.load_data_first import LoadDataTrain
from src.training.experiment import (log_run_to_mlflow, load_model_from_mlflow)



logging.basicConfig(
    level = logging.INFO,
    filename = Path(PATH_LOG).joinpath('tests_ppeline_logs.log'),
    filemode = "w",
    format = "%(asctime)s %(levelname)s %(message)s"
)


# ======================================================
# 1 Подготовка Loader
# ======================================================
loader = LoadDataTrain()

# ======================================================
# 2 Подготовка Scaler
# ======================================================
scaler_manager = Scaler()
scaler = scaler_manager.load_scaler(Path(PATH_SKALERS).joinpath("test_skaller.pkl"))

# ======================================================
# 3 Запуск Pipeline
# ======================================================
pipeline = Pipeline(
    path_data_dir = PATH_TRAIN_RAW,
    path_scaler = Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager = scaler_manager,
    loader = loader
        )

final_train, final_test, final_valid, final_anomal = pipeline.run()

logging.info(f"Results: final_train:{final_train}\n final_test:{final_test}\n final_valid:{final_valid}\n final_anomal:{final_anomal}\n")
# ======================================================
# 4 Проведение эксперимента
# ======================================================
