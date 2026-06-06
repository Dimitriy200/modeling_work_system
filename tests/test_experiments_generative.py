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
SEQ_LENGTH = 70  # Длина окна (например, 40 циклов)
STRIDE = 5      # Шаг сдвига. Меньше шаг = больше данных, но выше корреляция между окнами.

X_train_seq = processor.create_sequences(scaled_X_train, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_val_seq   = processor.create_sequences(scaled_X_val, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_test_seq  = processor.create_sequences(scaled_X_test, SEQ_LENGTH, STRIDE, FEATURE_COLS)

logging.info("=== FINAL DATA FORMS FOR VAE ===")
logging.info(f"X_train_seq: {X_train_seq.shape}")  # Ожидаем: (N_samples, 40, 16)
logging.info(f"X_val_seq:   {X_val_seq.shape}")
logging.info(f"X_test_seq:  {X_test_seq.shape}")

logging.info(f"Mean: {scaled_X_train.values.mean():.4f}")  # Должно быть ~0
logging.info(f"Std: {scaled_X_train.values.std():.4f}")    # Должно быть ~1


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
    print("CUDA not available! Training will run on CPU (very slow).")
    print("Check: NVIDIA drivers installed + CUDA Toolkit + PyTorch with CUDA support")
    device = torch.device("cpu")

print("=" * 50)
# ==========================================
# 1. Параметры обучения
# ==========================================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-5
WARMUP_EPOCHS = 80  # Эпохи для KL-Annealing (beta растет от 0 до 1)

CONTEXT_LEN = 40 

# Параметры архитектуры LSTM_VAE
HIDDEN_DIM = 32
LATENT_DIM = 8
N_LAYERS = 2

# ==========================================
# 2. Этап обучения
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
N_FEATURES = X_train_seq.shape[2]

model_vae = AdaptiveForecasting_VAE(
    input_dim=N_FEATURES,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    seq_len=SEQ_LENGTH,
    n_layers=N_LAYERS,
    context_len=CONTEXT_LEN,
    forecast_len=30,
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
# IV СБОР ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ
# ======================================================
models_dict = {
    'LSTM_VAE': model_vae,
    # 'Standard_AE': ae_standart,  # Раскомментируйте для сравнения
    # 'Compact_AE': ae_expansion
}

generation_metrics_df, generation_raw_df = run_generation_comparison_table(
    models=models_dict,
    X_real_test=X_test_seq,  # Реальные тестовые данные
    device=device,
    n_generate=50,          # Сколько семплов генерировать
    n_bootstrap=15,          # Бутстрап-итераций (уменьшите для быстрого теста)
    confidence_level=0.95,
    seed=42,
    return_raw=True
)

log_generation_report(generation_metrics_df)

# ======================================================
# V ИНФЕРЕНС
# ======================================================

N_CONTEXTS = 3
N_SAMPLES_COND = 5
SENSOR = "sensor measurement 2"

# Генерация (если ещё не выполнена)
real_contexts = torch.tensor(
    X_test_seq[:N_CONTEXTS, :CONTEXT_LEN, :], 
    dtype=torch.float32
).to(device)

with torch.no_grad():
    generated = model_vae.generate(real_contexts, n_samples=N_SAMPLES_COND, temperature=1.0)

logging.info(f"Сгенерировано условных траекторий: {generated.shape}")

# Визуализация условной генерации
plot_conditional_generation_inference(
    generated=generated,
    X_test_seq=X_test_seq,
    save_path=os.path.join(PATH_IMG, 'inference_conditional_generation.png'),
    context_len=CONTEXT_LEN,
    seq_len=SEQ_LENGTH,
    sensor_name=SENSOR,
    feature_names=FEATURE_COLS,
    n_samples=5,
    show_combined=True
)

# ==========================================
# 3.4 ИНФЕРЕНС: БЕЗУСЛОВНАЯ ГЕНЕРАЦИЯ
# ==========================================
N_SAMPLES_UNCOND = 20

with torch.no_grad():
    generated_uncond = model_vae.generate_from_noise(n_samples=N_SAMPLES_UNCOND, temperature=1.0)

logging.info(f"Generated unconditional trajectories: {generated_uncond.shape}")

# Визуализация безусловной генерации
plot_unconditional_generation_inference(
    generated_uncond=generated_uncond,
    save_path=os.path.join(PATH_IMG, 'inference_conditional_generation.png'),
    seq_len=SEQ_LENGTH,
    feature_idx=9,
    feature_names=FEATURE_COLS,
)

logging.info("=== Inference completed ===")