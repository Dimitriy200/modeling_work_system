import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_training_curves(
    history: Dict[str, list],
    save_path: str,
    warmup_epochs: Optional[int] = None,
    figsize: tuple = (15, 5)
) -> None:
    """
    Строит комплексные графики истории обучения VAE.
    
    Визуализирует:
      1. Общий Loss (Train vs Val) — сходимость модели
      2. Декомпозицию Train Loss: Reconstruction + KL Divergence
      3. (Опционально) Вертикальную линию warmup-фазы KL-Annealing
    
    Args:
        history: Словарь с ключами:
            - 'train_loss': список общих train-потерь по эпохам
            - 'val_loss': список общих val-потерь по эпохам
            - 'train_recon': список reconstruction loss по эпохам
            - 'train_kld': список KL divergence по эпохам
        save_path: Путь для сохранения PNG (например, PATH_IMG/training_curves.png).
        warmup_epochs: Количество эпох KL-Annealing (для визуализации фазы прогрева).
        figsize: Размер фигуры (width, height) в дюймах.
    """
    # === Валидация входных данных ===
    required_keys = ['train_loss', 'val_loss', 'train_recon', 'train_kld']
    missing_keys = [k for k in required_keys if k not in history]
    if missing_keys:
        raise ValueError(f"В history отсутствуют ключи: {missing_keys}. "
                         f"Ожидались: {required_keys}")
    
    epochs = len(history['train_loss'])
    if epochs < 2:
        logging.warning(f"Слишком мало эпох ({epochs}) для построения графиков.")
        return
    
    epoch_numbers = np.arange(1, epochs + 1)
    
    # === Создание фигуры с 3 сабплотами ===
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('LSTM-VAE Training History', fontsize=16, fontweight='bold', y=1.02)
    
    # ==========================================
    # График 1: Общий Loss (Train vs Val)
    # ==========================================
    ax1 = axes[0]
    ax1.plot(epoch_numbers, history['train_loss'], 
             label='Train Loss', color='#2E86AB', linewidth=2, marker='o', markersize=3)
    ax1.plot(epoch_numbers, history['val_loss'], 
             label='Val Loss', color='#A23B72', linewidth=2, marker='s', markersize=3)
    
    if warmup_epochs is not None and warmup_epochs > 0:
        ax1.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.7,
                    label=f'Warmup End (epoch {warmup_epochs})')
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Total Loss (MSE + β·KLD)', fontsize=11)
    ax1.set_title('Total Loss: Train vs Validation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1, epochs)
    
    # ==========================================
    # График 2: Декомпозиция Train Loss
    # ==========================================
    ax2 = axes[1]
    ax2.plot(epoch_numbers, history['train_recon'], 
             label='Reconstruction Loss (MSE)', color='#F18F01', linewidth=2, marker='o', markersize=3)
    ax2.plot(epoch_numbers, history['train_kld'], 
             label='KL Divergence', color='#C73E1D', linewidth=2, marker='s', markersize=3)
    
    if warmup_epochs is not None and warmup_epochs > 0:
        ax2.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.7,
                    label=f'Warmup End (epoch {warmup_epochs})')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss Value', fontsize=11)
    ax2.set_title('Train Loss Decomposition', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(1, epochs)
    
    # ==========================================
    # График 3: Соотношение Recon / KLD
    # ==========================================
    ax3 = axes[2]
    # Добавляем epsilon, чтобы избежать деления на ноль
    eps = 1e-8
    recon_arr = np.array(history['train_recon'])
    kld_arr = np.array(history['train_kld'])
    ratio = recon_arr / (kld_arr + eps)
    
    ax3.plot(epoch_numbers, ratio, color='#048A81', linewidth=2, marker='D', markersize=3,
             label='Recon / KLD Ratio')
    
    if warmup_epochs is not None and warmup_epochs > 0:
        ax3.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.7,
                    label=f'Warmup End (epoch {warmup_epochs})')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Ratio (Recon / KLD)', fontsize=11)
    ax3.set_title('Loss Components Balance', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(1, epochs)
    
    # === Финальная настройка и сохранение ===
    plt.tight_layout()
    
    # Создаем директорию, если её нет
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Training curves saved to: {save_path}")
    
    plt.show()
    plt.close(fig)
    
    # === Логирование итоговых значений ===
    logging.info("=== TRAINING HISTORY SUMMARY ===")
    logging.info(f"Total epochs trained: {epochs}")
    logging.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    logging.info(f"Final Val Loss:   {history['val_loss'][-1]:.4f}")
    logging.info(f"Final Recon Loss: {history['train_recon'][-1]:.4f}")
    logging.info(f"Final KLD Loss:   {history['train_kld'][-1]:.4f}")
    
    # Анализ сходимости
    if len(history['val_loss']) >= 5:
        last_5_val = history['val_loss'][-5:]
        if last_5_val[-1] > last_5_val[0]:
            logging.warning(
                f"⚠️ Val Loss вырос за последние 5 эпох "
                f"({last_5_val[0]:.4f} → {last_5_val[-1]:.4f}). "
                f"Возможно переобучение — рассмотрите EarlyStopping."
            )
        else:
            logging.info("✅ Val Loss steadily decreasing - the model converges.")
    
    # Анализ KL-Annealing
    if len(history['train_kld']) >= 2:
        if history['train_kld'][-1] < 1e-3:
            logging.warning(
                "⚠️ KL Divergence близка к нулю. Возможна проблема posterior collapse. "
                "Увеличьте WARMUP_EPOCHS или проверьте архитектуру."
            )
        else:
            logging.info(f"✅ KL Divergence is active ({history['train_kld'][-1]:.4f}) — "
                         f"latent space is used.")