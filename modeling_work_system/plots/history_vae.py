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
    Универсальная функция для визуализации истории обучения VAE.
    
    Автоматически определяет тип модели по ключам в history:
      - Обычный VAE: 'train_recon', 'train_kld'
      - Adaptive Forecasting VAE: 'train_context', 'train_forecast', 'alpha_history'
    
    Args:
        history: Словарь с историей обучения.
        save_path: Путь для сохранения PNG.
        warmup_epochs: Количество эпох KL-Annealing.
        figsize: Размер фигуры.
    """
    # === Определяем тип модели по ключам ===
    is_adaptive = 'train_context' in history and 'train_forecast' in history
    is_standard = 'train_recon' in history
    
    if not is_adaptive and not is_standard:
        raise ValueError(
            f"Неизвестный формат history. "
            f"Ожидались ключи 'train_recon' (стандартный VAE) "
            f"или 'train_context'+'train_forecast' (Adaptive VAE)."
        )
    
    # Базовая проверка
    required_base = ['train_loss', 'val_loss', 'train_kld']
    missing = [k for k in required_base if k not in history]
    if missing:
        raise ValueError(f"В history отсутствуют обязательные ключи: {missing}")
    
    epochs = len(history['train_loss'])
    if epochs < 2:
        logging.warning(f"Слишком мало эпох ({epochs}) для построения графиков.")
        return
    
    epoch_numbers = np.arange(1, epochs + 1)
    
    # === Определяем количество сабплотов ===
    if is_adaptive:
        # 4 графика: Total, Context vs Forecast, KLD, Weights
        n_plots = 4
        figsize = (20, 5)
    else:
        # 3 графика: Total, Recon vs KLD, Ratio
        n_plots = 3
        figsize = figsize
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    title = 'Adaptive Forecasting VAE Training' if is_adaptive else 'LSTM-VAE Training History'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
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
    ax1.set_ylabel('Total Loss', fontsize=11)
    ax1.set_title('Total Loss: Train vs Validation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1, epochs)
    
    # ==========================================
    # График 2: Декомпозиция Train Loss
    # ==========================================
    ax2 = axes[1]
    
    if is_adaptive:
        # Adaptive VAE: Context vs Forecast
        ax2.plot(epoch_numbers, history['train_context'], 
                 label='Context Loss (α-weighted)', color='#F18F01', 
                 linewidth=2, marker='o', markersize=3)
        ax2.plot(epoch_numbers, history['train_forecast'], 
                 label='Forecast Loss (β-weighted)', color='#C73E1D', 
                 linewidth=2, marker='s', markersize=3)
        ax2.set_title('Train Loss: Context vs Forecast', fontsize=12, fontweight='bold')
    else:
        # Standard VAE: Recon vs KLD
        ax2.plot(epoch_numbers, history['train_recon'], 
                 label='Reconstruction Loss (MSE)', color='#F18F01', 
                 linewidth=2, marker='o', markersize=3)
        ax2.plot(epoch_numbers, history['train_kld'], 
                 label='KL Divergence', color='#C73E1D', 
                 linewidth=2, marker='s', markersize=3)
        ax2.set_title('Train Loss Decomposition', fontsize=12, fontweight='bold')
    
    if warmup_epochs is not None and warmup_epochs > 0:
        ax2.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.7,
                    label=f'Warmup End (epoch {warmup_epochs})')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss Value', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(1, epochs)
    
    # ==========================================
    # График 3: KL Divergence или Ratio
    # ==========================================
    ax3 = axes[2]
    
    if is_adaptive:
        # KL Divergence отдельно
        ax3.plot(epoch_numbers, history['train_kld'], 
                 color='#048A81', linewidth=2, marker='D', markersize=3,
                 label='KL Divergence')
        ax3.set_title('KL Divergence', fontsize=12, fontweight='bold')
    else:
        # Ratio Recon / KLD
        eps = 1e-8
        recon_arr = np.array(history['train_recon'])
        kld_arr = np.array(history['train_kld'])
        ratio = recon_arr / (kld_arr + eps)
        
        ax3.plot(epoch_numbers, ratio, color='#048A81', linewidth=2, marker='D', markersize=3,
                 label='Recon / KLD Ratio')
        ax3.set_title('Loss Components Balance', fontsize=12, fontweight='bold')
    
    if warmup_epochs is not None and warmup_epochs > 0:
        ax3.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.7,
                    label=f'Warmup End (epoch {warmup_epochs})')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(1, epochs)
    
    # ==========================================
    # График 4 (только для Adaptive): Динамика весов
    # ==========================================
    if is_adaptive and 'alpha_history' in history:
        ax4 = axes[3]
        ax4.plot(epoch_numbers, history['alpha_history'], 
                 label='α (Context Weight)', color='#2E86AB', 
                 linewidth=2, marker='o', markersize=3)
        ax4.plot(epoch_numbers, history['beta_history'], 
                 label='β (Forecast Weight)', color='#A23B72', 
                 linewidth=2, marker='s', markersize=3)
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, 
                    label='Balance (0.5)')
        
        if warmup_epochs is not None and warmup_epochs > 0:
            ax4.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.7,
                        label=f'Warmup End (epoch {warmup_epochs})')
        
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Weight Value', fontsize=11)
        ax4.set_title('Adaptive Weights Evolution', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(1, epochs)
        ax4.set_ylim(0, 1)
    
    # === Финальная настройка и сохранение ===
    plt.tight_layout()
    
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
    logging.info(f"Final KLD Loss:   {history['train_kld'][-1]:.4f}")
    
    if is_adaptive:
        logging.info(f"Final Context Loss: {history['train_context'][-1]:.4f}")
        logging.info(f"Final Forecast Loss: {history['train_forecast'][-1]:.4f}")
        logging.info(f"Final α (context weight): {history['alpha_history'][-1]:.3f}")
        logging.info(f"Final β (forecast weight): {history['beta_history'][-1]:.3f}")
    else:
        logging.info(f"Final Recon Loss: {history['train_recon'][-1]:.4f}")
    
    # Анализ сходимости
    if len(history['val_loss']) >= 5:
        last_5_val = history['val_loss'][-5:]
        if last_5_val[-1] > last_5_val[0]:
            logging.warning(
                f"Val Loss вырос за последние 5 эпох "
                f"({last_5_val[0]:.4f} → {last_5_val[-1]:.4f}). "
                f"Overfitting is possible - consider EarlyStopping."
            )
        else:
            logging.info("✅ Val Loss steadily decreasing - the model converges.")
    
    # Анализ KL-Annealing
    if len(history['train_kld']) >= 2:
        if history['train_kld'][-1] < 1e-3:
            logging.warning(
                "KL Divergence is close to zero. Possible posterior collapse issue. "
                "Increase WARMUP_EPOCHS or check the architecture."
            )
        else:
            logging.info(f"✅ KL Divergence is active ({history['train_kld'][-1]:.4f}) — "
                         f"latent space is used.")
    
    # Анализ адаптивных весов (только для Adaptive VAE)
    if is_adaptive and 'alpha_history' in history:
        alpha_start = history['alpha_history'][0]
        alpha_end = history['alpha_history'][-1]
        beta_start = history['beta_history'][0]
        beta_end = history['beta_history'][-1]
        
        logging.info(f"\n=== ADAPTIVE WEIGHTS ANALYSIS ===")
        logging.info(f"α: {alpha_start:.3f} → {alpha_end:.3f} (Δ = {alpha_end - alpha_start:+.3f})")
        logging.info(f"β: {beta_start:.3f} → {beta_end:.3f} (Δ = {beta_end - beta_start:+.3f})")
        
        if abs(alpha_end - alpha_start) < 0.05:
            logging.info("Веса почти не изменились — баланс стабилен.")
        elif alpha_end > beta_end:
            logging.info("Модель фокусируется на восстановлении контекста (α > β).")
        else:
            logging.info("Модель фокусируется на прогнозе будущего (β > α).")