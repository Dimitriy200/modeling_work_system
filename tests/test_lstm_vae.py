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

from modeling_work_system.models.VAE.ts_vae import TimeSeriesForecastingVAE
from modeling_work_system.models.VAE.ts_1_vae import TimeSeriesIterativeVAE
from modeling_work_system.models.VAE.transformer_vae import TimeSeriesTransformerVAE


from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.metrics.generation_metrics import (
    run_generation_comparison_table,
    log_generation_report
)
from modeling_work_system.plots.inference_plot import plot_inference_results, plot_inference_multi_features
from modeling_work_system.plots.pocess_data_plots import plot_sensor_smoothing

from modeling_work_system.metrics.statistic_compare import paired_t_test

from modeling_work_system.config import (
    PATH_MODELS,
    PATH_LOG,
    PATH_SKALERS,
    PATH_IMG,

    PATH_TRAIN_RAW,
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

SAVE_MODEL = False
MODEL_NAME = "model_lstm_vae_v2_2.pth"

# ------------------------------
# ПАРАМЕТРЫ ОКОН
# ------------------------------
SEQ_LENGTH = 10                 # Длина окна 
STRIDE = 1                      # Шаг сдвига.
PAST_STEPS = 5                  # Первая часть окна - прошлое

# ------------------------------
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ------------------------------
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001 #5e-5
WARMUP_EPOCHS = 10  # Эпохи для KL-Annealing (beta растет от 0 до 1)
CONTEXT_LEN = 5
FORECAST_LEN = CONTEXT_LEN
KL_MINIMUM = 0.5 #0.15

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

marking_df = processor.marking_norm_anom_by_quantile(no_null_df)
splited_dataframes = processor.split_by_engine_train_test_val(dataframe=marking_df)

logging.info(f"X_train_anom = {splited_dataframes["X_train_anom"]}")
logging.info(f"X_val_anom = {splited_dataframes["X_val_anom"]}")
logging.info(f"X_test_anom = {splited_dataframes["X_test_anom"]}")

# ------------------------------
# Обучение Scaler
# ------------------------------
FEATURE_COLS = raw_df.columns.tolist()
std_scaler = scaler_manager.fit_scaler(splited_dataframes["X_train_norm"], FEATURE_COLS) # Обучаем Scaller только на нормальных и TRAIN данных !!!

# ------------------------------
# Проводим сглаживание 
# ------------------------------
df_norm_smoothing = {
    "Train": processor.smoothing_by_engine(df=splited_dataframes['X_train_norm'], sensor_columns=FEATURE_COLS),
    "Val": processor.smoothing_by_engine(df=splited_dataframes['X_val_norm'], sensor_columns=FEATURE_COLS),
    "Test": processor.smoothing_by_engine(df=splited_dataframes['X_test_norm'], sensor_columns=FEATURE_COLS)
}

df_anom_smoothing = {
    "Train": processor.smoothing_by_engine(df=splited_dataframes['X_train_anom'], sensor_columns=FEATURE_COLS),
    "Val": processor.smoothing_by_engine(df=splited_dataframes['X_val_anom'], sensor_columns=FEATURE_COLS),
    "Test": processor.smoothing_by_engine(df=splited_dataframes['X_test_anom'], sensor_columns=FEATURE_COLS)
}

# Смотрим результат сглаживания
CHECK_SENSORS = ["sensor measurement 2", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",]
for sensor in CHECK_SENSORS:
    plot_sensor_smoothing(
        df_raw=splited_dataframes['X_train_norm'],
        df_smoothed=df_norm_smoothing["Train"],
        engine_id = 1,
        feature_name = sensor
    )

for sensor in CHECK_SENSORS:
    plot_sensor_smoothing(
        df_raw=splited_dataframes['X_train_anom'],
        df_smoothed=df_anom_smoothing["Train"],
        engine_id = 1,
        feature_name = sensor
    )


# ------------------------------
# Применение Scaler
# ------------------------------
# Без сглаживания
X_scaled_norm = {
    "Train": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train_norm'], FEATURE_COLS),
    "Val": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val_norm'], FEATURE_COLS),
    "Test": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test_norm'], FEATURE_COLS) 
}

X_scaled_anom = {
    "Train": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_train_anom'], FEATURE_COLS),
    "Test": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_val_anom'], FEATURE_COLS),
    "Val": scaler_manager.apply_scaler(std_scaler, splited_dataframes['X_test_anom'], FEATURE_COLS)
}
logging.info(f"Mean of features: {X_scaled_norm['Train'].mean().mean():.4f}")
logging.info(f"Std of features: {X_scaled_norm['Test'].std().mean():.4f}")
logging.info(f"Std of features: {X_scaled_norm['Val'].std().mean():.4f}")

logging.info(f"Mean of features: {X_scaled_anom['Train'].mean().mean():.4f}")
logging.info(f"Std of features: {X_scaled_anom['Test'].std().mean():.4f}")
logging.info(f"Std of features: {X_scaled_anom['Val'].std().mean():.4f}")

# Со сглаживанием
X_scaled_norm_smoothing = {
    "Train": scaler_manager.apply_scaler(std_scaler, df_norm_smoothing['Train'], FEATURE_COLS),
    "Val": scaler_manager.apply_scaler(std_scaler, df_norm_smoothing['Val'], FEATURE_COLS),
    "Test": scaler_manager.apply_scaler(std_scaler, df_norm_smoothing['Test'], FEATURE_COLS) 
}

X_scaled_anom_smoothing = {
    "Train": scaler_manager.apply_scaler(std_scaler, df_anom_smoothing['Train'], FEATURE_COLS),
    "Test": scaler_manager.apply_scaler(std_scaler, df_anom_smoothing['Val'], FEATURE_COLS),
    "Val": scaler_manager.apply_scaler(std_scaler, df_anom_smoothing['Test'], FEATURE_COLS)
}

logging.info(f"Mean of features: {X_scaled_norm_smoothing['Train'].mean().mean():.4f}")
logging.info(f"Std of features: {X_scaled_norm_smoothing['Test'].std().mean():.4f}")
logging.info(f"Std of features: {X_scaled_norm_smoothing['Val'].std().mean():.4f}")

logging.info(f"Mean of features: {X_scaled_anom_smoothing['Train'].mean().mean():.4f}")
logging.info(f"Std of features: {X_scaled_anom_smoothing['Test'].std().mean():.4f}")
logging.info(f"Std of features: {X_scaled_anom_smoothing['Val'].std().mean():.4f}")


# ------------------------------
# Создание последовательностей - Нарезка окон
# ------------------------------
logging.info("=== FINAL DATA FORMS FOR VAE ===")

# Без сглаживания
df_norm_scaled_sec = {
    "Train": processor.create_sequences(X_scaled_norm["Train"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val": processor.create_sequences(X_scaled_norm["Val"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test": processor.create_sequences(X_scaled_norm_smoothing["Test"], SEQ_LENGTH, STRIDE, FEATURE_COLS)
}

df_anom_scaled_sec = {
    "Train": processor.create_sequences(X_scaled_anom["Train"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val": processor.create_sequences(X_scaled_anom["Val"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test": processor.create_sequences(X_scaled_anom["Test"], SEQ_LENGTH, STRIDE, FEATURE_COLS)
}
logging.info(f"df_norm_scaled_sec: {df_norm_scaled_sec['Train'].shape}")
logging.info(f"df_anom_scaled_sec: {df_anom_scaled_sec['Train'].shape}")

# Со сглаживанием
df_norm_scaled_sec_smoothing = {
    "Train": processor.create_sequences(X_scaled_norm_smoothing["Train"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val": processor.create_sequences(X_scaled_norm_smoothing["Val"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test": processor.create_sequences(X_scaled_norm_smoothing["Test"], SEQ_LENGTH, STRIDE, FEATURE_COLS)
}

df_anom_scaled_sec_smoothing = {
    "Train": processor.create_sequences(X_scaled_anom_smoothing["Train"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Val": processor.create_sequences(X_scaled_anom_smoothing["Val"], SEQ_LENGTH, STRIDE, FEATURE_COLS),
    "Test": processor.create_sequences(X_scaled_anom_smoothing["Test"], SEQ_LENGTH, STRIDE, FEATURE_COLS)
}
logging.info(f"df_norm_scaled_sec: {df_norm_scaled_sec['Train'].shape}")
logging.info(f"df_anom_scaled_sec_smoothing: {df_anom_scaled_sec_smoothing['Train'].shape}")

logging.info(f"df_norm_scaled_sec['Train']  is: {type(df_norm_scaled_sec['Train'])}")


# ------------------------------
# Выделяем первую часть окна - прошлое.
# ------------------------------
df_norm_scaled_seq_past = {
    "Train": df_norm_scaled_sec["Train"][:, :PAST_STEPS],
    "Val": df_norm_scaled_sec["Val"][:, :PAST_STEPS],
    "Test": df_norm_scaled_sec["Test"][:, :PAST_STEPS]
}

df_anom_scaled_seq_past = {
    "Train": df_anom_scaled_sec["Train"][:, :PAST_STEPS],
    "Val": df_anom_scaled_sec["Val"][:, :PAST_STEPS],
    "Test": df_anom_scaled_sec["Test"][:, :PAST_STEPS]
}

logging.info(f"df_norm_scaled_seq_past['Train']:  {df_norm_scaled_seq_past['Train'].shape}")
logging.info(f"df_anom_scaled_seq_past['Train']:  {df_anom_scaled_seq_past['Train'].shape}")

df_norm_scaled_seq_past_smoothong_past = {
    "Train": df_norm_scaled_sec_smoothing["Train"][:, :PAST_STEPS],
    "Val": df_norm_scaled_sec_smoothing["Val"][:, :PAST_STEPS],
    "Test": df_norm_scaled_sec_smoothing["Test"][:, :PAST_STEPS]
}

df_anom_scaled_seq_smoothong_past = {
    "Train": df_anom_scaled_sec_smoothing["Train"][:, :PAST_STEPS],
    "Val": df_anom_scaled_sec_smoothing["Val"][:, :PAST_STEPS],
    "Test": df_anom_scaled_sec_smoothing["Test"][:, :PAST_STEPS]
}

logging.info(f"df_norm_scaled_seq_past_smoothong_past['Train']:  {df_norm_scaled_seq_past_smoothong_past['Train'].shape}")
logging.info(f"df_anom_scaled_seq_smoothong_past['Train']:  {df_anom_scaled_seq_smoothong_past['Train'].shape}")


# ------------------------------
#  Будущее - для итеративного инференса
# ------------------------------
df_norm_scaled_seq_future = {
    "Train": df_norm_scaled_sec["Train"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val": df_norm_scaled_sec["Val"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test": df_norm_scaled_sec["Test"][:, int(PAST_STEPS)][:, np.newaxis]
}

df_anom_scaled_seq_future = {
    "Train": df_anom_scaled_sec["Train"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val": df_anom_scaled_sec["Val"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test": df_anom_scaled_sec["Test"][:, int(PAST_STEPS)][:, np.newaxis]
}

logging.info(f"df_norm_scaled_seq_future['Train']:  {df_norm_scaled_seq_future['Train'].shape}")
logging.info(f"df_anom_scaled_seq_future['Train']:  {df_anom_scaled_seq_future['Train'].shape}")

df_norm_scaled_seq_past_smoothong_future = {
    "Train": df_norm_scaled_sec_smoothing["Train"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val": df_norm_scaled_sec_smoothing["Val"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test": df_norm_scaled_sec_smoothing["Test"][:, int(PAST_STEPS)][:, np.newaxis]
}

df_anom_scaled_seq_smoothong_future = {
    "Train": df_anom_scaled_sec_smoothing["Train"][:, int(PAST_STEPS)][:, np.newaxis],
    "Val": df_anom_scaled_sec_smoothing["Val"][:, int(PAST_STEPS)][:, np.newaxis],
    "Test": df_anom_scaled_sec_smoothing["Test"][:, int(PAST_STEPS)][:, np.newaxis]
}

logging.info(f"df_norm_scaled_seq_past_smoothong_future['Train']:  {df_norm_scaled_seq_past_smoothong_future['Train'].shape}")
logging.info(f"df_anom_scaled_seq_smoothong_future['Train']:  {df_anom_scaled_seq_smoothong_future['Train'].shape}")

# y_train_sec_future = X_train_seq[:, int(PAST_STEPS)][:, np.newaxis]
# y_val_sec_future = X_val_seq[:, int(PAST_STEPS)][:, np.newaxis]
# y_test_sec_future = X_test_seq[:, int(PAST_STEPS)][:, np.newaxis]

# logging.info(f"y_train_sec_future:  {y_train_sec_future.shape}")
# logging.info(f"y_val_sec_future:  {y_val_sec_future.shape}")
# logging.info(f"y_test_sec_future:  {y_test_sec_future.shape}")

# ------------------------------
# Берем середину окна. Это должно уменьшить разброс при генерации в начале будущего.
# ------------------------------
df_norm_scaled_seq_ls = {
    "Train": df_norm_scaled_sec["Train"][:, PAST_STEPS - 1],
    "Val": df_norm_scaled_sec["Val"][:, PAST_STEPS - 1],
    "Test": df_norm_scaled_sec["Test"][:, PAST_STEPS - 1]
}

df_anom_scaled_seq_ls = {
    "Train": df_anom_scaled_sec["Train"][:, PAST_STEPS - 1],
    "Val": df_anom_scaled_sec["Val"][:, PAST_STEPS - 1],
    "Test": df_anom_scaled_sec["Test"][:, PAST_STEPS - 1]
}

logging.info(f"df_norm_scaled_seq_future['Train']:  {df_norm_scaled_seq_future['Train'].shape}")
logging.info(f"df_anom_scaled_seq_future['Train']:  {df_anom_scaled_seq_future['Train'].shape}")

df_norm_scaled_seq_past_smoothong_ls = {
    "Train": df_norm_scaled_sec_smoothing["Train"][:, PAST_STEPS - 1],
    "Val": df_norm_scaled_sec_smoothing["Val"][:, PAST_STEPS - 1],
    "Test": df_norm_scaled_sec_smoothing["Test"][:, PAST_STEPS - 1]
}

df_anom_scaled_seq_smoothong_ls = {
    "Train": df_anom_scaled_sec_smoothing["Train"][:, PAST_STEPS - 1],
    "Val": df_anom_scaled_sec_smoothing["Val"][:, PAST_STEPS - 1],
    "Test": df_anom_scaled_sec_smoothing["Test"][:, PAST_STEPS - 1]
}

logging.info(f"df_norm_scaled_seq_past_smoothong_future['Train']:  {df_norm_scaled_seq_past_smoothong_future['Train'].shape}")
logging.info(f"df_anom_scaled_seq_smoothong_future['Train']:  {df_anom_scaled_seq_smoothong_future['Train'].shape}")



# X_train_seq_ls = X_train_seq[:, PAST_STEPS - 1]
# X_val_seq_ls = X_val_seq[:, PAST_STEPS - 1]
# X_test_seq_ls = X_test_seq[:, PAST_STEPS - 1]

# logging.info(f"X_train_seq_ls:  {X_train_seq_ls.shape}")
# logging.info(f"X_val_seq_ls:  {X_val_seq_ls.shape}")
# logging.info(f"X_test_seq_ls:  {X_test_seq_ls.shape}")

# N_FEATURES = X_train_seq.shape[2]



# ======================================================
# II ОБУЧЕНИЕ МОДЕЛЕЙ
# ======================================================
history = model.fit(
    x_train=torch.FloatTensor(df_anom_scaled_seq_past["Train"]),                    # Сырое прошлое
    last_steps_train=torch.FloatTensor(df_anom_scaled_seq_smoothong_ls["Train"]),   # Сглаженная граница
    y_train=torch.FloatTensor(df_anom_scaled_seq_smoothong_future["Train"]),        #Сглаженное будущее
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    tau=KL_MINIMUM,
    verbose_step = 5,
)

plot_vae_training_history(history, save_path=os.path.join(PATH_IMG, 'plot_histore_vae_v2_2.png'))

# ------------------------------
# Сохраняем модель
# ------------------------------
if SAVE_MODEL:
    torch.save(model.state_dict(), os.path.join(PATH_MODELS, MODEL_NAME))


# ======================================================
# III ИНФЕРЕНС
# ======================================================
# ------------------------------
# НА НОРМЕ БЕЗ СГЛАЖИВАНИЯ
# ------------------------------
gen_scenarios_norm = model.inference(
    x_past=torch.FloatTensor(df_norm_scaled_seq_past["Val"]), 
    # last_known_step=torch.FloatTensor(X_val_seq_ls),
    horizon=10,
)
# ------------------------------
# НА АНОМАЛИИ
# ------------------------------

# ------------------------------
# Рисуем графиги инференса
# ------------------------------
num_engines_to_plot = 3 
for engine_idx in range(num_engines_to_plot):
    logging.info(f"Drawing and saving a graph for window (engine) No.{engine_idx}...")
    
    # 1. Извлекаем реальные данные (10, 26) для текущего двигателя
    y_true_single = df_norm_scaled_sec["Val"][engine_idx]
    
    # 2. Извлекаем сгенерированные сценарии (10, 26) конкретно для этого двигателя
    # Заходим в каждый из сэмплированных вариантов будущего и берем строку [engine_idx]
    single_engine_scenarios = [scenario[engine_idx] for scenario in gen_scenarios_norm]
    
    # 3. Формируем уникальное имя файла для каждого двигателя (например, engine_0.png, engine_1.png...)
    filename = f'plot_inference_lstm_vae_engine_{engine_idx}.png'
    current_save_path = os.path.join(PATH_IMG, filename)
    
    # 4. Вызываем функцию отрисовки (код внутри inference_plot.py менять не нужно, 
    # так как туда поступает чистая двухмерная матрица для одного двигателя)
    plot_inference_multi_features(
        y_true=y_true_single,
        scenarios=single_engine_scenarios,
        feature_indices=[6, 13],
        feature_names=["Sensor 2", "Sensor 9"],
        save_path=current_save_path
    )



# ======================================================
# IV СБОР СТАТИСТИЧЕССКИХ ДАННЫХ
# ======================================================
