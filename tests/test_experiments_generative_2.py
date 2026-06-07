import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


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
from modeling_work_system.models.VAE.lstm_vae import LSTM_VAE
from modeling_work_system.models.VAE.forecast_vae import AdaptiveForecasting_VAE
from modeling_work_system.models.VAE.conditional_lstm_vae import Conditional_LSTM_VAE
from modeling_work_system.models.VAE.ts_vae import TimeSeriesForecastingVAE


from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.metrics.generation_metrics import (
    run_generation_comparison_table,
    log_generation_report
)
from modeling_work_system.plots.generation_plots import (
    plot_conditional_generation_inference,
    plot_unconditional_generation_inference
)

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

from modeling_work_system.plots.history_vae_2 import plot_vae_training_history
from modeling_work_system.plots.vae_evaluation import evaluate_and_plot_vae

# ======================================================
# ПОДГОТОВКА ПЕРЕМЕННЫХ
# ======================================================
loader = LoadDataTrain()
scaler_manager = Scaler()
processor = Preprocess()
metrics = ExperimentMetric()

SEQ_LENGTH = 10                 # Длина окна 
STRIDE = 1                      # Шаг сдвига.
PAST_STEPS = 5                  # Первая часть окна - прошлое

# ПАРАМЕТРЫ ОБУЧЕНИЯ
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001 #5e-5
# WARMUP_EPOCHS = 10  # Эпохи для KL-Annealing (beta растет от 0 до 1)

CONTEXT_LEN = 5
FORECAST_LEN = CONTEXT_LEN
KL_MINIMUM = 0.15

# ПАРАМЕТРЫ АРХИТКТУРЫ LSTM_VAE
FEATURE_DIM = 26
LATENT_DIM = 4
N_LAYERS = 2

model = TimeSeriesForecastingVAE(
    feature_dim = FEATURE_DIM,
    latent_dim = LATENT_DIM
)

# ======================================================
# ПОИСК CUDA
# ======================================================
print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    device = torch.device("cuda")
else:
    print("CUDA not available! Training will run on CPU (very slow).")
    print("Check: NVIDIA drivers installed + CUDA Toolkit + PyTorch with CUDA support")
    device = torch.device("cpu")

print("=" * 50)


# ======================================================
# I ПОДГОТОВКА ДАННЫХ
# ======================================================
raw_df = loader.data_raw_load(PATH_TRAIN_RAW)
no_null_df = processor.delete_nan(raw_df)

marking_df = processor.marking_norm_anom_by_quantile(no_null_df)
splited_dataframes = processor.split_by_engine_train_test_val(dataframe=marking_df)

logging.info(f"X_train_anom = {splited_dataframes["X_train_anom"]}")
logging.info(f"X_val_anom = {splited_dataframes["X_val_anom"]}")
logging.info(f"X_test_anom = {splited_dataframes["X_test_anom"]}")

# 2.1 Обучение и применение Scaler
FEATURE_COLS = raw_df.columns.tolist()
std_scaler = scaler_manager.fit_scaler(splited_dataframes["X_train_norm"], FEATURE_COLS) # Обучаем Scaller только на нормальных и TRAIN данных !!!

scaled_X_train = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train_norm'], FEATURE_COLS)
scaled_X_val = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val_norm'], FEATURE_COLS)
scaled_X_test = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test_norm'], FEATURE_COLS)

scaled_X_train_anom = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train_anom'], FEATURE_COLS)
scaled_X_val_anom = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val_anom'], FEATURE_COLS)
scaled_X_test_anom = scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test_anom'], FEATURE_COLS)

logging.info(f"Mean of features: {scaled_X_train.mean().mean():.4f}")
logging.info(f"Std of features: {scaled_X_val.std().mean():.4f}")
logging.info(f"Std of features: {scaled_X_test.std().mean():.4f}")
logging.info(f"Std of features: {scaled_X_train_anom.std().mean():.4f}")
logging.info(f"Std of features: {scaled_X_val_anom.std().mean():.4f}")
logging.info(f"Std of features: {scaled_X_test_anom.std().mean():.4f}")


# 2.2 Создание последовательностей
logging.info("=== FINAL DATA FORMS FOR VAE ===")

# Нарезаем окна.
X_train_seq = processor.create_sequences(scaled_X_train, SEQ_LENGTH, STRIDE, FEATURE_COLS) 
X_val_seq   = processor.create_sequences(scaled_X_val, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_test_seq  = processor.create_sequences(scaled_X_test, SEQ_LENGTH, STRIDE, FEATURE_COLS)

logging.info(f"X_train_seq: {X_train_seq.shape}")  # Ожидаем: (N_samples, 40, 16)
logging.info(f"X_val_seq:   {X_val_seq.shape}")
logging.info(f"X_test_seq:  {X_test_seq.shape}")
logging.info(f"Mean: {scaled_X_train.values.mean():.4f}")  # Должно быть ~0
logging.info(f"Std: {scaled_X_train.values.std():.4f}")    # Должно быть ~1
logging.info(f"X_train_seq  is: {type(X_train_seq)}")    # Должно быть ~1


# Выделяем первую часть окна - прошлое.
X_train_seq_past = X_train_seq[:, :PAST_STEPS]
X_val_seq_past = X_val_seq[:, :PAST_STEPS]
X_test_seq_past = X_test_seq[:, :PAST_STEPS]

logging.info(f"X_train_seq_past:  {X_train_seq_past.shape}")
logging.info(f"X_val_seq_past:  {X_val_seq_past.shape}")
logging.info(f"X_test_seq_past:  {X_test_seq_past.shape}")

# Возможно нужно нарезать и будущее

# Берем середину окна. Это должно уменьшить разброс при генерации в начале будущего.
X_train_seq_ls = X_train_seq[:, PAST_STEPS - 1]
X_val_seq_ls = X_val_seq[:, PAST_STEPS - 1]
X_test_seq_ls = X_test_seq[:, PAST_STEPS - 1]

logging.info(f"X_train_seq_ls:  {X_train_seq_ls.shape}")
logging.info(f"X_val_seq_ls:  {X_val_seq_ls.shape}")
logging.info(f"X_test_seq_ls:  {X_test_seq_ls.shape}")

N_FEATURES = X_train_seq.shape[2]

# ======================================================
# II ОБУЧЕНИЕ МОДЕЛЕЙ
# ======================================================
history = model.fit(
    X_train=torch.FloatTensor(X_train_seq_past),
    last_steps_train=torch.FloatTensor(X_train_seq_ls),
    y_train=torch.FloatTensor(X_train_seq),
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    tau=KL_MINIMUM,
    verbose_step = 5
)
plot_vae_training_history(history)

# ======================================================
# III ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ [ОБУЧЕНИЕ / КЛАССИФИКАЦИЯ / ВОССТАНОВЛЕНИЕ ДАННЫХ]
# ======================================================



# ======================================================
# IV СБОР ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ
# ======================================================


# ======================================================
# V ИНФЕРЕНС
# ======================================================

