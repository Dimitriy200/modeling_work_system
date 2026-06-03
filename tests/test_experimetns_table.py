import pandas as pd
import numpy as np
import logging

import sys
import os
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from modeling_work_system.preprocessing.preprocessing import Preprocess
from modeling_work_system.pipeline.pipeline_fit import PipelineFit
from modeling_work_system.preprocessing.scaler import Scaler
from modeling_work_system.preprocessing.load_data_first import LoadDataTrain

from modeling_work_system.models.autoencoders.autoencoder import AutoEncoder
from modeling_work_system.models.core_models.autoencoders import STANDART_AE, EXPANSION_AE

from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.metrics.compose_table_metrics import (
    run_reconstruction_comparison_table, 
    run_classification_comparison_table)
from modeling_work_system.metrics.statistic_compare import paired_t_test

from modeling_work_system.config import (
    PATH_LOG,
    PATH_SKALERS,
    PATH_IMG,

    PATH_TRAIN_RAW,
    PATH_TRAIN_ADD_RAW,

    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_REPO_OWNER,
    MLFLOW_REPO_NAME,
    MLFLOW_REPO_TOKEN,
    MLFLOW_REPO_PASSWORD
)

from modeling_work_system.plots.history import plot_training_curves


# ======================================================
# ПОДГООВКА ПЕРЕМЕННЫХ
# ======================================================
loader = LoadDataTrain()
scaler_manager = Scaler()
processor = Preprocess()
metrics = ExperimentMetric()

ae_standart = AutoEncoder(model_core=STANDART_AE)
ae_expansion = AutoEncoder(model_core=EXPANSION_AE)


# ======================================================
# I ПОДГОТОВКА ДАННЫХ
# ======================================================
raw_df = loader.data_raw_load(PATH_TRAIN_RAW)
no_null_df = processor.delete_nan(raw_df)

marking_df = processor.marking_norm_anom(no_null_df)
splited_dataframes = processor.split_by_engine_train_test_val(dataframe=marking_df)

# 2.1 Обучение и применение Scaler
cols = raw_df.columns.tolist()
std_scaler = scaler_manager.fit_scaler(splited_dataframes["X_train"], cols) # Обучаем Scaller только на нормальных и TRAIN данных !!!

scaled_X_train = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train'], cols)
scaled_X_val = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val'], cols)
scaled_X_test = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test'], cols)


# ======================================================
# II ОБУЧЕНИЕ МОДЕЛЕЙ
# ======================================================
train_info = ae_standart.fit(
    X_train=scaled_X_train,
    X_val=scaled_X_val,
    X_test=scaled_X_test,
    Y_val=splited_dataframes['y_val'],
    epochs=3)


# ======================================================
# III ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ [ОБУЧЕНИЕ / КЛАССИФИКАЦИЯ / ВОССТАНОВЛЕНИЕ ДАННЫХ]
# ======================================================
plot_training_curves(
    history=train_info["history"],
    save_path=os.path.join(PATH_IMG, "standart_ae_history.png"))