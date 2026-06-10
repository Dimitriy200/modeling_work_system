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

from modeling_work_system.models.VAE.lstm_vae import TimeSeriesIterativeVAE

from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.metrics.generation_metrics import (
    run_generation_comparison_table,
    log_generation_report
)
from modeling_work_system.plots.inference_plot import plot_inference_results, plot_inference_multi_features
from modeling_work_system.plots.pocess_data_plots import plot_sensor_smoothing, plot_sensor_stress_testing
from modeling_work_system.plots.generation_plots import plot_recursive_lifetime_forecast

from modeling_work_system.metrics.statistic_compare import paired_t_test

from modeling_work_system.config import (
    PATH_MODELS,
    PATH_LOG,
    PATH_SKALERS,
    PATH_IMG,
    PATH_TRAIN_PROCESSED,

    PATH_TRAIN_RAW,
)

from modeling_work_system.plots.history_vae_2 import plot_vae_training_history
from modeling_work_system.plots.vae_evaluation import evaluate_and_plot_vae
from modeling_work_system.metrics.metrics_vae import evaluate_model_noise_robustness_advanced


# ======================================================
# ПОДГОТОВКА ПЕРЕМЕННЫХ
# ======================================================


loader = LoadDataTrain()
scaler_manager = Scaler()
processor = Preprocess()
metrics = ExperimentMetric()

# ------------------------------
# ОБЩИЕ ПАРАМЕТРЫ
# ------------------------------
PATH_IMG_LSTM = os.path.join(PATH_IMG, "lstm_vae")
PATH_TRAIN_PROCESSED_LSTM = os.path.join(PATH_TRAIN_PROCESSED, "experiments")

SAVE_MODEL = False              # Сохранение модели в файл
LOAD_MODEL = False

MODEL_NAME = "lstm_vae"         # Имя модели при сохранении
MODEL_VERSION = "v2"

# ------------------------------
# ПАРАМЕТРЫ ОБРАБОТКИ ДАННЫХ
# ------------------------------
N_LAST_ANOM = 50
QUANTILE = 0.90

DROP_RATE = 0.1
NOISE_RATE=0.1

# ------------------------------
# ПАРАМЕТРЫ ОКОН
# ------------------------------
SEQ_LENGTH = 20                 # Длина окна 
STRIDE = 1                      # Шаг сдвига.
PAST_STEPS = 10                  # Первая часть окна - прошлое

# ------------------------------
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ------------------------------
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.001 #5e-5
# WARMUP_EPOCHS = 10  # Эпохи для KL-Annealing (beta растет от 0 до 1)
CONTEXT_LEN = 5
FORECAST_LEN = CONTEXT_LEN
KL_MINIMUM = 0.1 #0.15

# ------------------------------
# ПАРАМЕТРЫ АРХИТКТУРЫ МОДЕЛИ
# ------------------------------
FEATURE_DIM = 26
LATENT_DIM = 4
N_LAYERS = 2

model = TimeSeriesIterativeVAE(
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

marking_df = processor.marking_norm_anom_by_quantile(no_null_df, quantile=QUANTILE)
# marking_df = processor.marking_norm_anom(no_null_df, n_anom = N_LAST_ANOM)
splited_dataframes = processor.split_by_engine_train_test_val(dataframe=marking_df)

logging.info(f"X_train_anom = {splited_dataframes["X_train_anom"].shape}")
logging.info(f"X_val_anom = {splited_dataframes["X_val_anom"].shape}")
logging.info(f"X_test_anom = {splited_dataframes["X_test_anom"].shape}")


# ------------------------------
# Обучение Scaler
# ------------------------------
FEATURE_COLS = raw_df.columns.tolist()
SENSOES_SETTINGS_COLS = [col for col in FEATURE_COLS if col not in ['unit number', 'time in cycles']]

std_scaler = scaler_manager.fit_scaler(splited_dataframes["X_train_norm"], FEATURE_COLS) # Обучаем Scaller только на нормальных и TRAIN данных !!!


# ------------------------------
# Применение Scaler
# ------------------------------
df_scaled = {
    "Train_norm": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train_norm'], FEATURE_COLS),
    "Val_norm": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val_norm'], FEATURE_COLS),
    "Test_norm": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test_norm'], FEATURE_COLS),

    "Train_anom": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train_anom'], FEATURE_COLS),
    "Val_anom": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val_anom'], FEATURE_COLS),
    "Test_anom": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test_anom'], FEATURE_COLS)
}
logging.info(f"Std of features: {df_scaled['Train_norm'].std().mean():.4f}")


# ------------------------------
# Добавление шума
# ------------------------------
df_noiseing = {
    # DROP - ПРОПУСК
    "Train_norm_drop": processor.apply_stress_to_data(df_scaled['Train_norm'], SENSOES_SETTINGS_COLS, noise_type="drop", drop_rate=DROP_RATE),
    "Val_norm_drop": processor.apply_stress_to_data(df_scaled['Val_norm'], SENSOES_SETTINGS_COLS, noise_type="drop", drop_rate=DROP_RATE),
    "Test_norm_drop": processor.apply_stress_to_data(df_scaled['Test_norm'], SENSOES_SETTINGS_COLS, noise_type="drop", drop_rate=DROP_RATE),

    "Train_anom_drop": processor.apply_stress_to_data(df_scaled['Train_anom'], SENSOES_SETTINGS_COLS, noise_type="drop", drop_rate=DROP_RATE),
    "Val_anom_drop": processor.apply_stress_to_data(df_scaled['Val_anom'], SENSOES_SETTINGS_COLS, noise_type="drop", drop_rate=DROP_RATE),
    "Test_anom_drop": processor.apply_stress_to_data(df_scaled['Test_anom'], SENSOES_SETTINGS_COLS, noise_type="drop", drop_rate=DROP_RATE),

    # NOISE - БЕЛЫЙ ШУМ
    "Train_norm_noise": processor.apply_stress_to_data(df_scaled['Train_norm'], SENSOES_SETTINGS_COLS, noise_type="noise", noise_level=NOISE_RATE),
    "Val_norm_noise": processor.apply_stress_to_data(df_scaled['Val_norm'], SENSOES_SETTINGS_COLS, noise_type="noise", noise_level=NOISE_RATE),
    "Test_norm_noise": processor.apply_stress_to_data(df_scaled['Test_norm'], SENSOES_SETTINGS_COLS, noise_type="noise", noise_level=NOISE_RATE),

    "Train_anom_noise": processor.apply_stress_to_data(df_scaled['Train_anom'], SENSOES_SETTINGS_COLS, noise_type="noise", noise_level=NOISE_RATE),
    "Val_anom_noise": processor.apply_stress_to_data(df_scaled['Val_anom'], SENSOES_SETTINGS_COLS, noise_type="noise", noise_level=NOISE_RATE),
    "Test_anom_noise": processor.apply_stress_to_data(df_scaled['Test_anom'], SENSOES_SETTINGS_COLS, noise_type="noise", noise_level=NOISE_RATE),

    # BOTH - ВСЕ ВИДЫ ШУМА ВМЕСТЕ
    "Train_norm_both": processor.apply_stress_to_data(df_scaled['Train_norm'], SENSOES_SETTINGS_COLS, noise_type="both", noise_level=NOISE_RATE, drop_rate=DROP_RATE),
    "Val_norm_both": processor.apply_stress_to_data(df_scaled['Val_norm'], SENSOES_SETTINGS_COLS, noise_type="both", noise_level=NOISE_RATE, drop_rate=DROP_RATE),
    "Test_norm_both": processor.apply_stress_to_data(df_scaled['Test_norm'], SENSOES_SETTINGS_COLS, noise_type="both", noise_level=NOISE_RATE, drop_rate=DROP_RATE),

    "Train_anom_both": processor.apply_stress_to_data(df_scaled['Train_anom'], SENSOES_SETTINGS_COLS, noise_type="both", noise_level=NOISE_RATE, drop_rate=DROP_RATE),
    "Val_anom_both": processor.apply_stress_to_data(df_scaled['Val_anom'], SENSOES_SETTINGS_COLS, noise_type="both", noise_level=NOISE_RATE, drop_rate=DROP_RATE),
    "Test_anom_both": processor.apply_stress_to_data(df_scaled['Test_anom'], SENSOES_SETTINGS_COLS, noise_type="both", noise_level=NOISE_RATE, drop_rate=DROP_RATE)
}
logging.info(f"df_noiseing[Train_norm_noise] = {df_noiseing["Train_norm_noise"]}")

# СМОТРИМ РЕЗУЛЬТАТЫ ДОБАВЛЕНИЯ ШУМА
# CHECK_SENSORS = ["sensor measurement 2", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",]
# for sensor in CHECK_SENSORS:
plot_sensor_stress_testing(
    array_clean=df_scaled['Train_norm'],
    array_stressed=df_noiseing["Train_norm_drop"],
    feature_name="sensor measurement 2",
    group_number=2,
    save_path=os.path.join(PATH_IMG, "proc_data", "plot_stressed_drop.png")
)

plot_sensor_stress_testing(
    array_clean=df_scaled['Train_norm'],
    array_stressed=df_noiseing["Train_norm_noise"],
    feature_name = "sensor measurement 2",
    group_number=2,
    save_path=os.path.join(PATH_IMG, "proc_data", "plot_stressed_noise.png")
)

plot_sensor_stress_testing(
    array_clean=df_scaled['Train_norm'],
    array_stressed=df_noiseing["Train_norm_both"],
    feature_name = "sensor measurement 2",
    group_number=2,
    save_path=os.path.join(PATH_IMG, "proc_data", "plot_stressed_both.png")
)


# ------------------------------
# Создание последовательностей - Нарезка окон
# ------------------------------
logging.info("=== FINAL DATA FORMS FOR VAE ===")

# Без сглаживания
df_sequences = {
    # ЧИСТЫЕ - ДЛЯ ОБУЧЕНИЯ
    "Scaled_Train_norm": processor.create_sequences(df_scaled["Train_norm"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Scaled_Val_norm": processor.create_sequences(df_scaled["Val_norm"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Scaled_Test_norm": processor.create_sequences(df_scaled["Test_norm"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    "Scaled_Train_anom": processor.create_sequences(df_scaled["Train_anom"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Scaled_Val_anom": processor.create_sequences(df_scaled["Val_anom"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Scaled_Test_anom": processor.create_sequences(df_scaled["Test_anom"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    # ЗАШУМЛЕННЫЕ - ДЛЯ ИНФЕРЕНСА
    # ПРОПУСКИ
    "Train_norm_drop": processor.create_sequences(df_noiseing["Train_norm_drop"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val_norm_drop": processor.create_sequences(df_noiseing["Val_norm_drop"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test_norm_drop": processor.create_sequences(df_noiseing["Test_norm_drop"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    "Train_anom_drop": processor.create_sequences(df_noiseing["Train_anom_drop"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val_anom_drop": processor.create_sequences(df_noiseing["Val_anom_drop"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test_anom_drop": processor.create_sequences(df_noiseing["Test_anom_drop"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    # БЕЛЫЙ ШУМ
    "Train_norm_noise": processor.create_sequences(df_noiseing["Train_norm_noise"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val_norm_noise": processor.create_sequences(df_noiseing["Val_norm_noise"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test_norm_noise": processor.create_sequences(df_noiseing["Test_norm_noise"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    "Train_anom_noise": processor.create_sequences(df_noiseing["Train_anom_noise"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val_anom_noise": processor.create_sequences(df_noiseing["Val_anom_noise"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test_anom_noise": processor.create_sequences(df_noiseing["Test_anom_noise"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    # BМЕСТЕ
    "Train_norm_both": processor.create_sequences(df_noiseing["Train_norm_both"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val_norm_both": processor.create_sequences(df_noiseing["Val_norm_both"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test_norm_both": processor.create_sequences(df_noiseing["Test_norm_both"], SEQ_LENGTH, STRIDE, FEATURE_COLS),

    "Train_anom_both": processor.create_sequences(df_noiseing["Train_anom_both"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val_anom_both": processor.create_sequences(df_noiseing["Val_anom_both"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test_anom_both": processor.create_sequences(df_noiseing["Test_anom_both"], SEQ_LENGTH, STRIDE, FEATURE_COLS)
}

logging.info(f"df_sequences: {df_sequences['Scaled_Train_norm'].shape}")


# ------------------------------
# Выделяем первую часть окна - прошлое.
# ------------------------------
df_sequences_past = {
    # ЧИСТЫЕ - ДЛЯ ОБУЧЕНИЯ
    "Scaled_Train_norm":    df_sequences["Scaled_Train_norm"][:, :PAST_STEPS],
    "Scaled_Val_norm":      df_sequences["Scaled_Val_norm"][:, :PAST_STEPS],
    "Scaled_Test_norm":     df_sequences["Scaled_Test_norm"][:, :PAST_STEPS],

    "Scaled_Train_anom":    df_sequences["Scaled_Train_anom"][:, :PAST_STEPS],
    "Scaled_Val_anom":      df_sequences["Scaled_Val_anom"][:, :PAST_STEPS],
    "Scaled_Test_anom":     df_sequences["Scaled_Test_anom"][:, :PAST_STEPS],
    

    # ЗАШУМЛЕННЫЕ - ДЛЯ ИНФЕРЕНСА
    # ПРОПУСКИ
    "Train_norm_drop":      df_sequences["Train_norm_drop"][:, :PAST_STEPS],
    "Val_norm_drop":        df_sequences["Val_norm_drop"][:, :PAST_STEPS],
    "Test_norm_drop":       df_sequences["Test_norm_drop"][:, :PAST_STEPS],

    "Train_anom_drop":      df_sequences["Train_anom_drop"][:, :PAST_STEPS],
    "Val_anom_drop":        df_sequences["Val_anom_drop"][:, :PAST_STEPS],
    "Test_anom_drop":       df_sequences["Test_anom_drop"][:, :PAST_STEPS],

    # БЕЛЫЙ ШУМ
    "Train_norm_noise":     df_sequences["Train_norm_noise"][:, :PAST_STEPS],
    "Val_norm_noise":       df_sequences["Val_norm_noise"][:, :PAST_STEPS],
    "Test_norm_noise":      df_sequences["Test_norm_noise"][:, :PAST_STEPS],

    "Train_anom_noise":     df_sequences["Train_anom_noise"][:, :PAST_STEPS],
    "Val_anom_noise":       df_sequences["Val_anom_noise"][:, :PAST_STEPS],
    "Test_anom_noise":      df_sequences["Test_anom_noise"][:, :PAST_STEPS],

    # BМЕСТЕ
    "Train_norm_both":      df_sequences["Train_norm_both"][:, :PAST_STEPS],
    "Val_norm_both":        df_sequences["Val_norm_both"][:, :PAST_STEPS],
    "Test_norm_both":       df_sequences["Test_norm_both"][:, :PAST_STEPS],

    "Train_anom_both":      df_sequences["Train_anom_both"][:, :PAST_STEPS],
    "Val_anom_both":        df_sequences["Val_anom_both"][:, :PAST_STEPS],
    "Test_anom_both":       df_sequences["Test_anom_both"][:, :PAST_STEPS]
}
logging.info(f"df_norm_scaled_seq_past['Train']:  {df_sequences_past['Scaled_Train_norm'].shape}")
logging.info(f"df_anom_scaled_seq_past['Train']:  {df_sequences_past['Train_norm_both'].shape}")

# ------------------------------
#  Будущее - для итеративного инференса
# ------------------------------
df_sequences_future = {
    # ЧИСТЫЕ - ДЛЯ ОБУЧЕНИЯ
    "Scaled_Train_norm":    df_sequences["Scaled_Train_norm"][:, int(PAST_STEPS)][:, np.newaxis],
    "Scaled_Val_norm":      df_sequences["Scaled_Val_norm"][:, int(PAST_STEPS)][:, np.newaxis],
    "Scaled_Test_norm":     df_sequences["Scaled_Test_norm"][:, int(PAST_STEPS)][:, np.newaxis],

    "Scaled_Train_anom":    df_sequences["Scaled_Train_anom"][:, int(PAST_STEPS)][:, np.newaxis],
    "Scaled_Val_anom":      df_sequences["Scaled_Val_anom"][:, int(PAST_STEPS)][:, np.newaxis],
    "Scaled_Test_anom":     df_sequences["Scaled_Test_anom"][:, int(PAST_STEPS)][:, np.newaxis],
    

    # ЗАШУМЛЕННЫЕ - ДЛЯ ИНФЕРЕНСА
    # ПРОПУСКИ
    "Train_norm_drop":      df_sequences["Train_norm_drop"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val_norm_drop":        df_sequences["Val_norm_drop"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test_norm_drop":       df_sequences["Test_norm_drop"][:, int(PAST_STEPS)][:, np.newaxis],

    "Train_anom_drop":      df_sequences["Train_anom_drop"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val_anom_drop":        df_sequences["Val_anom_drop"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test_anom_drop":       df_sequences["Test_anom_drop"][:, int(PAST_STEPS)][:, np.newaxis],

    # БЕЛЫЙ ШУМ
    "Train_norm_noise":     df_sequences["Train_norm_noise"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val_norm_noise":       df_sequences["Val_norm_noise"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test_norm_noise":      df_sequences["Test_norm_noise"][:, int(PAST_STEPS)][:, np.newaxis],

    "Train_anom_noise":     df_sequences["Train_anom_noise"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val_anom_noise":       df_sequences["Val_anom_noise"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test_anom_noise":      df_sequences["Test_anom_noise"][:, int(PAST_STEPS)][:, np.newaxis],

    # BМЕСТЕ
    "Train_norm_both":      df_sequences["Train_norm_both"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val_norm_both":        df_sequences["Val_norm_both"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test_norm_both":       df_sequences["Test_norm_both"][:, int(PAST_STEPS)][:, np.newaxis],

    "Train_anom_both":      df_sequences["Train_anom_both"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val_anom_both":        df_sequences["Val_anom_both"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test_anom_both":       df_sequences["Test_anom_both"][:, int(PAST_STEPS)][:, np.newaxis]
}
logging.info(f"df_norm_scaled_seq_past['Train']:  {df_sequences_future['Scaled_Train_norm'].shape}")
logging.info(f"df_anom_scaled_seq_past['Train']:  {df_sequences_future['Train_norm_both'].shape}")


# ------------------------------
# Берем середину окна. Это должно уменьшить разброс при генерации в начале будущего.
# ------------------------------
df_sequences_ls = {
    # ЧИСТЫЕ - ДЛЯ ОБУЧЕНИЯ
    "Scaled_Train_norm":    df_sequences["Scaled_Train_norm"][:, PAST_STEPS - 1],
    "Scaled_Val_norm":      df_sequences["Scaled_Val_norm"][:, PAST_STEPS - 1],
    "Scaled_Test_norm":     df_sequences["Scaled_Test_norm"][:, PAST_STEPS - 1],

    "Scaled_Train_anom":    df_sequences["Scaled_Train_anom"][:, PAST_STEPS - 1],
    "Scaled_Val_anom":      df_sequences["Scaled_Val_anom"][:, PAST_STEPS - 1],
    "Scaled_Test_anom":     df_sequences["Scaled_Test_anom"][:, PAST_STEPS - 1],
    

    # ЗАШУМЛЕННЫЕ - ДЛЯ ИНФЕРЕНСА
    # ПРОПУСКИ
    "Train_norm_drop":      df_sequences["Train_norm_drop"][:, PAST_STEPS - 1],
    "Val_norm_drop":        df_sequences["Val_norm_drop"][:, PAST_STEPS - 1],
    "Test_norm_drop":       df_sequences["Test_norm_drop"][:, PAST_STEPS - 1],

    "Train_anom_drop":      df_sequences["Train_anom_drop"][:, PAST_STEPS - 1],
    "Val_anom_drop":        df_sequences["Val_anom_drop"][:, PAST_STEPS - 1],
    "Test_anom_drop":       df_sequences["Test_anom_drop"][:, PAST_STEPS - 1],

    # БЕЛЫЙ ШУМ
    "Train_norm_noise":     df_sequences["Train_norm_noise"][:, PAST_STEPS - 1],
    "Val_norm_noise":       df_sequences["Val_norm_noise"][:, PAST_STEPS - 1],
    "Test_norm_noise":      df_sequences["Test_norm_noise"][:, PAST_STEPS - 1],

    "Train_anom_noise":     df_sequences["Train_anom_noise"][:, PAST_STEPS - 1],
    "Val_anom_noise":       df_sequences["Val_anom_noise"][:, PAST_STEPS - 1],
    "Test_anom_noise":      df_sequences["Test_anom_noise"][:, PAST_STEPS - 1],

    # BМЕСТЕ
    "Train_norm_both":      df_sequences["Train_norm_both"][:, PAST_STEPS - 1],
    "Val_norm_both":        df_sequences["Val_norm_both"][:, PAST_STEPS - 1],
    "Test_norm_both":       df_sequences["Test_norm_both"][:, PAST_STEPS - 1],

    "Train_anom_both":      df_sequences["Train_anom_both"][:, PAST_STEPS - 1],
    "Val_anom_both":        df_sequences["Val_anom_both"][:, PAST_STEPS - 1],
    "Test_anom_both":       df_sequences["Test_anom_both"][:, PAST_STEPS - 1]
}
logging.info(f"df_norm_scaled_seq_past['Train']:  {df_sequences_ls['Scaled_Train_norm'].shape}")
logging.info(f"df_anom_scaled_seq_past['Train']:  {df_sequences_ls['Train_norm_both'].shape}")


# ======================================================
# II ОБУЧЕНИЕ МОДЕЛЕЙ ИЛИ ЗАГРУЗКА ИЗ ФАЙЛА
# ======================================================
if LOAD_MODEL:
    model.load_state_dict(torch.load(os.path.join(PATH_MODELS, f"model_{MODEL_NAME}_{MODEL_VERSION}.pth"), map_location=device))
    model.to(device) # Переносим модель на видеокарту или процессор
    model.eval()
else:
    history = model.fit(
        x_train=torch.FloatTensor(df_sequences_past["Scaled_Train_norm"]),          # Чистое (незашумленное) прошлое Scaled_Train_norm
        last_steps_train=torch.FloatTensor(df_sequences_ls["Scaled_Train_norm"]),   # Чистая (незашумленная) граница
        y_train=torch.FloatTensor(df_sequences_future["Scaled_Train_norm"]),        # Чистое (незашумленное) будущее
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        tau=KL_MINIMUM,
        verbose_step = 5,
    )

    plot_vae_training_history(history, save_path=os.path.join(PATH_IMG_LSTM, f"plot_history_{MODEL_NAME}_{MODEL_VERSION}.png"))

# ------------------------------
# Сохраняем модель
# ------------------------------
if SAVE_MODEL:
    torch.save(model.state_dict(), os.path.join(PATH_MODELS, f"model_{MODEL_NAME}_{MODEL_VERSION}.pth"))


# ======================================================
# III ИНФЕРЕНС
# ======================================================
# ------------------------------
# НА НОРМЕ БЕЗ ЗАШУМЛЕНИЯ
# ------------------------------
gen_scenarios_norm = model.inference(
    x_past=torch.FloatTensor(df_sequences["Scaled_Val_norm"]), 
    # last_known_step=torch.FloatTensor(X_val_seq_ls),
    horizon=SEQ_LENGTH,
)

# Рисуем графиги инференса
num_engines_to_plot = 3
for engine_idx in range(num_engines_to_plot):
    logging.info(f"Drawing and saving a graph for window (engine) No.{engine_idx}...")
    
    # 1. Извлекаем реальные данные (10, 26) для текущего двигателя
    y_true_single = df_sequences["Scaled_Val_norm"][engine_idx]
    
    # 2. Извлекаем сгенерированные сценарии (10, 26) конкретно для этого двигателя
    # Заходим в каждый из сэмплированных вариантов будущего и берем строку [engine_idx]
    single_engine_scenarios = [scenario[engine_idx] for scenario in gen_scenarios_norm]
    
    # 3. Формируем уникальное имя файла для каждого двигателя (например, engine_0.png, engine_1.png...)
    current_save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_norm_{MODEL_NAME}_{MODEL_VERSION}_eng_{engine_idx}.png")
    
    # 4. Вызываем функцию отрисовки (код внутри inference_plot.py менять не нужно, 
    # так как туда поступает чистая двухмерная матрица для одного двигателя)
    plot_inference_multi_features(
        y_true=y_true_single,
        scenarios=single_engine_scenarios,
        plot_name=f"Инференс на норме без зашумления | двигатель {engine_idx}",
        feature_indices=[6, 13],
        feature_names=["Sensor 2", "Sensor 9"],
        save_path=current_save_path,
        past_steps=PAST_STEPS
    )

ENGINE_N = 0
unit_col = "unit number"

unique_engine_ids = df_scaled["Val_norm"][unit_col].unique()
target_scaled_id = unique_engine_ids[ENGINE_N - 1] 
single_engine_full_df = df_scaled["Val_norm"][df_scaled["Val_norm"][unit_col] == target_scaled_id]

plot_recursive_lifetime_forecast(
    model=model,
    start_x_past=torch.FloatTensor(df_sequences_past["Scaled_Val_norm"][ENGINE_N]),
    full_engine_df=single_engine_full_df,
    feature_idx=13,
    feature_name = "sensor measurement 9",
    num_scenarios = 3,
    save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_ft_norm_eng({ENGINE_N})_eng_{MODEL_NAME}.png")
)


# ------------------------------
# НА НОРМЕ С ЗАШУМЛЕНИЕМ (ПРОПУСКИ)
# ------------------------------
gen_scenarios_norm_drop = model.inference(
    x_past=torch.FloatTensor(df_sequences["Val_norm_drop"]), 
    # last_known_step=torch.FloatTensor(X_val_seq_ls),
    horizon=SEQ_LENGTH,
)

# Рисуем графики инференса
num_engines_to_plot = 3 
for engine_idx in range(num_engines_to_plot):
    logging.info(f"Drawing and saving a graph for window (engine) No.{engine_idx}...")
    
    # 1. Извлекаем реальные данные (10, 26) для текущего двигателя
    y_true_single = df_sequences["Val_norm_drop"][engine_idx]
    y_clean=df_sequences["Scaled_Val_norm"][engine_idx]
    
    # 2. Извлекаем сгенерированные сценарии (10, 26) конкретно для этого двигателя
    # Заходим в каждый из сэмплированных вариантов будущего и берем строку [engine_idx]
    single_engine_scenarios = [scenario[engine_idx] for scenario in gen_scenarios_norm_drop]
    
    # 3. Формируем уникальное имя файла для каждого двигателя (например, engine_0.png, engine_1.png...)
    current_save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_norm_drop_{MODEL_NAME}_{MODEL_VERSION}_eng_{engine_idx}.png")
    
    # 4. Вызываем функцию отрисовки (код внутри inference_plot.py менять не нужно, 
    # так как туда поступает чистая двухмерная матрица для одного двигателя)
    plot_inference_multi_features(
        y_true=y_true_single,
        y_clean=y_clean,
        scenarios=single_engine_scenarios,
        plot_name="Инференс на норме С зашумлением (пропуски)",
        feature_indices=[6, 13],
        feature_names=["Sensor 2", "Sensor 9"],
        save_path=current_save_path,
        past_steps=PAST_STEPS
    )

plot_recursive_lifetime_forecast(
    model=model,
    start_x_past=torch.FloatTensor(df_sequences_past["Val_norm_drop"][ENGINE_N]),
    full_engine_df=single_engine_full_df,
    feature_idx=13,
    feature_name = "sensor measurement 9",
    num_scenarios = 3,
    save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_ft_norm_drop_eng({ENGINE_N})_eng_{MODEL_NAME}.png")
)

# ------------------------------
# НА НОРМЕ С ЗАШУМЛЕНИЕМ (БЕЛЫЙ ШУМ)
# ------------------------------
gen_scenarios_norm_noise = model.inference(
    x_past=torch.FloatTensor(df_sequences["Val_norm_noise"]), 
    # last_known_step=torch.FloatTensor(X_val_seq_ls),
    horizon=SEQ_LENGTH,
)

# Рисуем графики инференса
num_engines_to_plot = 3 
for engine_idx in range(num_engines_to_plot):
    logging.info(f"Drawing and saving a graph for window (engine) No.{engine_idx}...")
    
    # 1. Извлекаем реальные данные (10, 26) для текущего двигателя
    y_true_single = df_sequences["Val_norm_noise"][engine_idx]
    y_clean=df_sequences["Scaled_Val_norm"][engine_idx]
    
    # 2. Извлекаем сгенерированные сценарии (10, 26) конкретно для этого двигателя
    # Заходим в каждый из сэмплированных вариантов будущего и берем строку [engine_idx]
    single_engine_scenarios = [scenario[engine_idx] for scenario in gen_scenarios_norm_noise]
    
    # 3. Формируем уникальное имя файла для каждого двигателя (например, engine_0.png, engine_1.png...)
    current_save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_norm_noise_{MODEL_NAME}_{MODEL_VERSION}_eng_{engine_idx}.png")
    
    # 4. Вызываем функцию отрисовки (код внутри inference_plot.py менять не нужно, 
    # так как туда поступает чистая двухмерная матрица для одного двигателя)
    plot_inference_multi_features(
        y_true=y_true_single,
        y_clean=y_clean,
        scenarios=single_engine_scenarios,
        plot_name="Инференс на норме С зашумлением (белый шум)",
        feature_indices=[6, 13],
        feature_names=["Sensor 2", "Sensor 9"],
        save_path=current_save_path,
        past_steps=PAST_STEPS
    )

plot_recursive_lifetime_forecast(
    model=model,
    start_x_past=torch.FloatTensor(df_sequences_past["Val_norm_noise"][ENGINE_N]),
    full_engine_df=single_engine_full_df,
    feature_idx=13,
    feature_name = "sensor measurement 9",
    num_scenarios = 3,
    save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_ft_norm_noise_eng({ENGINE_N})_eng_{MODEL_NAME}.png")
)

# ------------------------------
# НА НОРМЕ С ЗАШУМЛЕНИЕМ (БЕЛЫЙ ШУМ + ПРОПУСКИ)
# ------------------------------
gen_scenarios_norm_both = model.inference(
    x_past=torch.FloatTensor(df_sequences["Val_norm_both"]), 
    # last_known_step=torch.FloatTensor(X_val_seq_ls),
    horizon=SEQ_LENGTH,
)

# Рисуем графики инференса
num_engines_to_plot = 3 
for engine_idx in range(num_engines_to_plot):
    logging.info(f"Drawing and saving a graph for window (engine) No.{engine_idx}...")
    
    # 1. Извлекаем реальные данные (10, 26) для текущего двигателя
    y_true_single = df_sequences["Val_norm_both"][engine_idx]
    y_clean=df_sequences["Scaled_Val_norm"][engine_idx]
    
    # 2. Извлекаем сгенерированные сценарии (10, 26) конкретно для этого двигателя
    # Заходим в каждый из сэмплированных вариантов будущего и берем строку [engine_idx]
    single_engine_scenarios = [scenario[engine_idx] for scenario in gen_scenarios_norm_both]
    
    # 3. Формируем уникальное имя файла для каждого двигателя (например, engine_0.png, engine_1.png...)
    current_save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_norm_both_{MODEL_NAME}_{MODEL_VERSION}_eng_{engine_idx}.png")
    
    # 4. Вызываем функцию отрисовки (код внутри inference_plot.py менять не нужно, 
    # так как туда поступает чистая двухмерная матрица для одного двигателя)
    plot_inference_multi_features(
        y_true=y_true_single,
        y_clean=y_clean,
        scenarios=single_engine_scenarios,
        plot_name="Инференс на норме С зашумлением (белый шум и пропуски)",
        feature_indices=[6, 13],
        feature_names=["Sensor 2", "Sensor 9"],
        save_path=current_save_path,
        past_steps=PAST_STEPS
    )

plot_recursive_lifetime_forecast(
    model=model,
    start_x_past=torch.FloatTensor(df_sequences_past["Val_norm_both"][ENGINE_N]),
    full_engine_df=single_engine_full_df,
    feature_idx=13,
    feature_name = "sensor measurement 9",
    num_scenarios = 3,
    save_path = os.path.join(PATH_IMG_LSTM, f"plot_inf_ft_norm_noise_eng({ENGINE_N})_eng_{MODEL_NAME}.png")
)

# ------------------------------
# НА АНОМАЛИИ БЕЗ СГЛАЖИВАНИЯ
# ------------------------------

# ------------------------------
# НА АНОМАЛИИ СО СГЛАЖИВАНИЕМ
# ------------------------------

# ======================================================
# IV СБОР СТАТИСТИЧЕССКИХ ДАННЫХ
# ======================================================

logging.info(f"DROP_RATE = {DROP_RATE}")
logging.info(f"NOISE_RATE = {NOISE_RATE}")

# ------------------------------
# НА НОРМЕ БЕЗ ЗАШУМЛЕНИЯ
# ------------------------------
evaluate_model_noise_robustness_advanced(
    model=model,
    X_stressed_past=df_sequences_past["Scaled_Test_norm"],
    Y_clean_full=df_sequences["Scaled_Test_norm"]
)

# ------------------------------
# НА НОРМЕ С ЗАШУМЛЕНИЕМ (ПРОПУСКИ)
# ------------------------------
evaluate_model_noise_robustness_advanced(
    model=model,
    X_stressed_past=df_sequences_past["Test_norm_drop"],
    Y_clean_full=df_sequences["Scaled_Test_norm"]
)

# ------------------------------
# НА НОРМЕ С ЗАШУМЛЕНИЕМ (БЕЛЫЙ ШУМ)
# ------------------------------
evaluate_model_noise_robustness_advanced(
    model=model,
    X_stressed_past=df_sequences_past["Test_norm_noise"],
    Y_clean_full=df_sequences["Scaled_Test_norm"]
)

# ------------------------------
# НА НОРМЕ С ЗАШУМЛЕНИЕМ (БЕЛЫЙ ШУМ + ПРОПУСКИ)
# ------------------------------
evaluate_model_noise_robustness_advanced(
    model=model,
    X_stressed_past=df_sequences_past["Test_norm_both"],
    Y_clean_full=df_sequences["Scaled_Test_norm"]
)