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
from modeling_work_system.models.VAE.conditional_lstm_vae import Conditional_LSTM_VAE

from modeling_work_system.mlflowservice.mlflowservice import Mlflowservice
from modeling_work_system.metrics.metrics import ExperimentMetric
from modeling_work_system.metrics.aemetrics import AEMetricResult

from modeling_work_system.metrics.generation_metrics import (
    run_generation_comparison_table,
    log_generation_report
)

from modeling_work_system.metrics.vae_inference import (
    classify_anomalies_by_percentile,
    plot_classification_results
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
SEQ_LENGTH = 40  # Длина окна (например, 40 циклов)
STRIDE = 10      # Шаг сдвига. Меньше шаг = больше данных, но выше корреляция между окнами.

X_train_seq = processor.create_sequences(scaled_X_train, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_val_seq   = processor.create_sequences(scaled_X_val, SEQ_LENGTH, STRIDE, FEATURE_COLS)
X_test_seq  = processor.create_sequences(scaled_X_test, SEQ_LENGTH, STRIDE, FEATURE_COLS)

X_test_anom_seq  = processor.create_anomaly_sequences(scaled_X_test_anom, SEQ_LENGTH, STRIDE, FEATURE_COLS)

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
    print("CUDA not available! Training will run on CPU (very slow).")
    print("Check: NVIDIA drivers installed + CUDA Toolkit + PyTorch with CUDA support")
    device = torch.device("cpu")

print("=" * 50)
# ==========================================
# 1. Параметры обучения
# ==========================================
BATCH_SIZE = 16
EPOCHS = 50
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

model_vae = Conditional_LSTM_VAE(
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
CONTEXT_LEN = 20

# 3.1 Условная генерация (по контексту)
# Берем 3 реальных контекста из тестовых данных
n_contexts = 3
real_contexts = torch.tensor(
    X_test_seq[:n_contexts, :CONTEXT_LEN, :], 
    dtype=torch.float32
).to(device)

# Генерируем 5 вариантов продолжения для каждого контекста
with torch.no_grad():
    generated = model_vae.generate(real_contexts, n_samples=5, temperature=1.0)
# generated shape: (5, 3, 40, N_FEATURES)

logging.info(f"Сгенерировано условных траекторий: {generated.shape}")

# 3.2 Безусловная генерация (из чистого шума)
generated_uncond = model_vae.generate_from_noise(n_samples=20, temperature=1.0)
# generated_uncond shape: (20, 40, N_FEATURES)

logging.info(f"Сгенерировано безусловных траекторий: {generated_uncond.shape}")

# 3.3 Визуализация
# Выбираем один сенсор для отображения (например, sensor_21 - это индекс 23 в FEATURE_COLS)
# Проверьте актуальный индекс в вашем FEATURE_COLS
feature_idx = 20  # sensor_20, например

fig, axes = plt.subplots(n_contexts, 3, figsize=(18, 4 * n_contexts))
fig.suptitle('Conditional Generation: Context → Future', fontsize=16, fontweight='bold')

for i in range(n_contexts):
    # Левая колонка: реальная траектория
    ax1 = axes[i, 0]
    real_full = X_test_seq[i, :, feature_idx]
    ax1.plot(range(CONTEXT_LEN), real_full[:CONTEXT_LEN], 'bo-', 
             markersize=6, label='Context (seen)', linewidth=2)
    ax1.plot(range(CONTEXT_LEN, SEQ_LENGTH), real_full[CONTEXT_LEN:], 'g^-', 
             markersize=4, label='Future (actual)', linewidth=1.5, alpha=0.8)
    ax1.axvline(x=CONTEXT_LEN-0.5, color='red', linestyle='--', alpha=0.7, 
                label='Context boundary')
    ax1.set_title(f'Engine #{i+1}: Real Data')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Sensor Value (scaled)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Правая колонка: сгенерированные варианты
    ax2 = axes[i, 1]
    ax2.plot(range(CONTEXT_LEN), real_full[:CONTEXT_LEN], 'bo-', 
             markersize=6, label='Context (seen)', linewidth=2)
    
    # Рисуем все 5 сгенерированных вариантов
    for j in range(generated.shape[0]):
        gen_seq = generated[j, i, :, feature_idx]
        ax2.plot(range(CONTEXT_LEN, SEQ_LENGTH), gen_seq[CONTEXT_LEN:], 
                 'r-', alpha=0.4, linewidth=1, label='Generated' if j==0 else None)
    
    # Среднее значение
    gen_mean = generated[:, i, :, feature_idx].mean(axis=0)
    ax2.plot(range(CONTEXT_LEN, SEQ_LENGTH), gen_mean[CONTEXT_LEN:], 
             'm-', linewidth=3, label='Generated Mean', zorder=10)
    
    ax2.axvline(x=CONTEXT_LEN-0.5, color='red', linestyle='--', alpha=0.7)
    ax2.set_title(f'Engine #{i+1}: 5 Generated Variants')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Sensor Value (scaled)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Третья колонка: ошибка (реальность vs среднее генерации)
    ax3 = axes[i, 2]
    ax3.plot(range(SEQ_LENGTH), real_full, 'b-', linewidth=2, label='Real')
    ax3.plot(range(SEQ_LENGTH), gen_mean, 'm--', linewidth=2, label='Generated Mean')
    ax3.fill_between(range(SEQ_LENGTH), 
                     gen_mean - generated[:, i, :, feature_idx].std(axis=0),
                     gen_mean + generated[:, i, :, feature_idx].std(axis=0),
                     alpha=0.2, color='magenta', label='Std')
    ax3.set_title(f'Engine #{i+1}: Real vs Generated')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Sensor Value (scaled)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PATH_IMG, 'inference_conditional_generation.png'), dpi=300)
plt.show()

# 3.4 Визуализация безусловной генерации
fig, axes = plt.subplots(4, 5, figsize=(20, 12))
fig.suptitle('Unconditional Generation: 20 Synthetic Trajectories', 
             fontsize=16, fontweight='bold')

for i in range(4):
    for j in range(5):
        idx = i * 5 + j
        ax = axes[i, j]
        ax.plot(generated_uncond[idx, :, feature_idx], 'b-', linewidth=1.5)
        ax.set_title(f'Trajectory #{idx+1}')
        ax.set_xlabel('Time')
        if j == 0:
            ax.set_ylabel('Sensor Value')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PATH_IMG, 'inference_unconditional_generation.png'), dpi=300)
plt.show()

logging.info("Инференс завершен!")


# ==========================================
# 3.3 КЛАССИФИКАЦИЯ АНОМАЛИЙ
# ==========================================
logging.info("\n" + "=" * 60)
logging.info("ЗАПУСК ИНФЕРЕНСА И КЛАССИФИКАЦИИ")
logging.info("=" * 60)

# Запуск классификации
inference_results = classify_anomalies_by_percentile(
    model=model_vae,
    X_test_norm=X_test_seq,
    X_test_anom=X_test_anom_seq,
    device=device,
    percentile_threshold=95.0,
    batch_size=64
)

# Визуализация результатов
CLASSIFICATION_PLOT_PATH = os.path.join(PATH_IMG, "classification_results.png")
plot_classification_results(
    inference_results=inference_results,
    save_path=CLASSIFICATION_PLOT_PATH,
    figsize=(18, 5)
)

# Сохранение метрик в CSV
import json
metrics_save_path = os.path.join(PATH_SKALERS, "classification_metrics.json")
with open(metrics_save_path, 'w') as f:
    json.dump(inference_results['metrics'], f, indent=2)
logging.info(f"Метрики сохранены: {metrics_save_path}")