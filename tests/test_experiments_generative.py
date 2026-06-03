import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader

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

from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.metrics.compose_table_metrics import (
    run_reconstruction_comparison_table, 
    run_classification_comparison_table,
    log_summary_report)
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

from modeling_work_system.plots.history_vae import plot_training_curves
from modeling_work_system.plots.vae_evaluation import evaluate_and_plot_vae

# ======================================================
# ПОДГОТОВКА ПЕРЕМЕННЫХ
# ======================================================
loader = LoadDataTrain()
scaler_manager = Scaler()
processor = Preprocess()
metrics = ExperimentMetric()


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
SEQ_LENGTH = 40  # Длина окна (например, 40 циклов)
STRIDE = 10      # Шаг сдвига. Меньше шаг = больше данных, но выше корреляция между окнами.

X_train_seq = processor.create_sequences(scaled_X_train, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_val_seq   = processor.create_sequences(scaled_X_val, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_test_seq  = processor.create_sequences(scaled_X_test, SEQ_LENGTH, STRIDE, FEATURE_COLS)

logging.info("=== FINAL DATA FORMS FOR VAE ===")
logging.info(f"X_train_seq: {X_train_seq.shape}")  # Ожидаем: (N_samples, 40, 16)
logging.info(f"X_val_seq:   {X_val_seq.shape}")
logging.info(f"X_test_seq:  {X_test_seq.shape}")


# ======================================================
# II ОБУЧЕНИЕ МОДЕЛЕЙ
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
    print("⚠️ CUDA not available! Training will run on CPU (very slow).")
    print("Check: NVIDIA drivers installed + CUDA Toolkit + PyTorch with CUDA support")
    device = torch.device("cpu")

print("=" * 50)
# ==========================================
# 1. Параметры обучения
# ==========================================
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10  # Эпохи для KL-Annealing (beta растет от 0 до 1)

# Параметры архитектуры LSTM_VAE
HIDDEN_DIM = 64
LATENT_DIM = 16
N_LAYERS = 2

# ==========================================
# 2. Этап обучения
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
N_FEATURES = X_train_seq.shape[2]

model_vae = LSTM_VAE(
    input_dim=N_FEATURES,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    seq_len=SEQ_LENGTH,
    n_layers=N_LAYERS
)

logging.info(f"Model initialized. Total parameters: {sum(p.numel() for p in model_vae.parameters()):,}")

# 3. Запуск обучения через метод fit
# Модель сама создаст DataLoader, запустит цикл и вернет историю
training_history = model_vae.fit(
    X_train=X_train_seq,
    X_val=X_val_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    warmup_epochs=WARMUP_EPOCHS,
    device=device
)

# 4. Сохранение весов обученной модели
# MODEL_SAVE_PATH = os.path.join(PATH_SKALERS, "lstm_vae_model.pth")
# torch.save(model_vae.state_dict(), MODEL_SAVE_PATH)
# logging.info(f"Model weights saved to: {MODEL_SAVE_PATH}")


# ======================================================
# III ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ [ОБУЧЕНИЕ / КЛАССИФИКАЦИЯ / ВОССТАНОВЛЕНИЕ ДАННЫХ]
# ======================================================

evaluation_metrics = plot_training_curves(
    history=training_history,
    save_path=os.path.join(PATH_IMG, "history_vae.png"),
    warmup_epochs=WARMUP_EPOCHS,  # Передаем, чтобы отметить фазу прогрева
    figsize=(18, 5)
)


# ======================================================
# IV Сбор экспериментальных данных
# ======================================================
