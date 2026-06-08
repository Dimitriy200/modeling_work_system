
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

from typing import Optional, List


def plot_conditional_generation_inference(
    generated: np.ndarray,
    X_test_seq: np.ndarray,
    save_path: str,
    context_len: int,
    seq_len: int,
    sensor_name: str,              # ← ОБЯЗАТЕЛЬНЫЙ параметр
    feature_names: List[str],      # ← ОБЯЗАТЕЛЬНЫЙ параметр
    n_samples: int = 5,
    show_combined: bool = True
) -> None:
    """
    Визуализирует результаты условной генерации.
    Каждый пример (двигатель) отображается на ОТДЕЛЬНОМ холсте.
    """
    import matplotlib.pyplot as plt
    
    # ==========================================
    # НАХОДИМ ИНДЕКС СЕНСОРА ПО ИМЕНИ
    # ==========================================
    if sensor_name.startswith('s') and sensor_name[1:].isdigit():
        sensor_num = sensor_name[1:]
        target_name = f'sensor measurement {sensor_num}'
    elif sensor_name.isdigit():
        target_name = f'sensor measurement {sensor_name}'
    else:
        target_name = sensor_name
    
    try:
        feature_idx = feature_names.index(target_name)
        y_label = target_name
        logging.info(f"✓ Using sensor: {target_name} (index {feature_idx})")
    except ValueError:
        available = [name for name in feature_names if 'sensor' in name.lower()]
        raise ValueError(
            f"Sensor '{target_name}' not found in feature_names.\n"
            f"Available sensors: {available}"
        )
    
    # ==========================================
    # ВАЛИДАЦИЯ
    # ==========================================
    if generated.ndim != 4:
        raise ValueError(f"generated должен иметь форму (n_samples, n_contexts, seq_len, features)")
    
    n_gen_samples, n_contexts, gen_seq_len, n_features = generated.shape
    
    if len(X_test_seq) < n_contexts:
        n_contexts = len(X_test_seq)
    
    # === ОДИН ГРАФИК (убрали правую колонку) ===
    for i in range(n_contexts):
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        fig.suptitle(
            f'Engine #{i+1}: Conditional Generation\n'
            f'Sensor: {y_label} | Samples: {n_gen_samples}',
            fontsize=14, fontweight='bold', y=1.02
        )
        
        real_full = X_test_seq[i, :, feature_idx]
        gen_mean = generated[:, i, :, feature_idx].mean(axis=0)
        gen_std = generated[:, i, :, feature_idx].std(axis=0)
        
        # === КОНТЕКСТ + ГЕНЕРАЦИЯ ===
        ax.plot(
            range(context_len), real_full[:context_len], 
            color='#232D48', marker='o', linestyle='-', 
            markersize=3, linewidth=1.5, alpha=0.7,
            label='Context (real)'
        )
        
        ax.plot(
            range(context_len, seq_len), real_full[context_len:], 
            color='#1D6522', marker='o', linestyle='-', 
            markersize=3, linewidth=2, alpha=0.5,
            label='Future (actual)'
        )
        
        # Все сгенерированные варианты
        for j in range(n_gen_samples):
            gen_seq = generated[j, i, :, feature_idx]
            ax.plot(
                range(context_len, seq_len), gen_seq[context_len:], 
                color="#6D3B84", linewidth=0.8, alpha=0.5
            )
            steps = range(context_len, seq_len)
            ax.scatter(
                steps[::2], gen_seq[context_len:][::2], 
                color="#6D3B84", s=15, alpha=0.6, zorder=5
            )
        
        # Среднее сгенерированное
        ax.plot(
            range(context_len, seq_len), gen_mean[context_len:], 
            'm-', linewidth=3, label='Generated Mean', zorder=10
        )
        ax.scatter(
            range(context_len, seq_len)[::2], gen_mean[context_len:][::2], 
            color='magenta', s=40, marker='D', label='Generated Points', zorder=11
        )
        
        # Доверительный интервал
        ax.fill_between(
            range(seq_len), gen_mean - gen_std, gen_mean + gen_std,
            alpha=0.2, color='magenta', label='±1 Std'
        )
        
        # Граница контекста
        ax.axvline(
            x=context_len - 0.5, color='red', linestyle='--', 
            alpha=0.7, linewidth=2, label='Context boundary'
        )
        
        ax.set_title('Context + Generated Forecast', fontsize=11)
        ax.set_xlabel('Time Step (cycles)')
        ax.set_ylabel(y_label)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        base_name = save_path.replace('.png', '')
        current_save_path = f"{base_name}_engine_{i+1}.png"
        
        os.makedirs(os.path.dirname(current_save_path), exist_ok=True)
        plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved: {current_save_path}")
        
        plt.show()
        plt.close(fig)
    
    logging.info(f"All {n_contexts} conditional generation plots saved.")




def plot_unconditional_generation_inference(
    generated_uncond: np.ndarray,
    save_path: str,
    seq_len: int,
    feature_idx: int = 1,
    feature_names: Optional[List[str]] = None,
    n_cols: int = 5,
    figsize_per_cell: tuple = (4, 3)
) -> None:
    """
    Визуализирует результаты безусловной генерации (инференс).
    
    Строит сетку из сгенерированных траекторий.
    
    Args:
        generated_uncond: Сгенерированные данные формы (n_samples, seq_len, features).
        save_path: Путь для сохранения PNG.
        seq_len: Длина последовательности.
        feature_idx: Индекс признака для визуализации.
        feature_names: Названия признаков (для подписи оси Y).
        n_cols: Количество колонок в сетке.
        figsize_per_cell: Размер одной ячейки (width, height).
    """
    import matplotlib.pyplot as plt
    
    # Валидация
    if generated_uncond.ndim != 3:
        raise ValueError(
            f"generated_uncond должен иметь форму (n_samples, seq_len, features), "
            f"получено: {generated_uncond.shape}"
        )
    
    n_samples, gen_seq_len, n_features = generated_uncond.shape
    
    if gen_seq_len != seq_len:
        raise ValueError(
            f"Несовпадение seq_len: в generated={gen_seq_len}, ожидалось={seq_len}"
        )
    
    # Подпись оси Y
    if feature_names is not None and feature_idx < len(feature_names):
        y_label = feature_names[feature_idx]
    else:
        y_label = f'Feature {feature_idx} (scaled)'
    
    # Расчёт сетки
    n_rows = (n_samples + n_cols - 1) // n_cols
    figsize = (figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Обработка граничных случаев
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(
        f'Unconditional Generation: {n_samples} Synthetic Trajectories\n'
        f'Feature: {y_label}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    # Цветовая палитра
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Определяем общий диапазон осей для единообразия
    y_min = generated_uncond[:, :, feature_idx].min()
    y_max = generated_uncond[:, :, feature_idx].max()
    y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
    
    for idx in range(n_samples):
        i, j = idx // n_cols, idx % n_cols
        ax = axes[i, j]
        
        ax.plot(
            generated_uncond[idx, :, feature_idx], 
            color=colors[idx % 10], 
            linewidth=1.5, marker='o', markersize=2
        )
        ax.set_title(f'Trajectory #{idx+1}', fontsize=10)
        ax.set_xlabel('Time Step (cycles)')
        if j == 0:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        # Единый диапазон осей для сравнения
        ax.set_ylim([y_min - y_margin, y_max + y_margin])
    
    # Скрываем пустые subplot'ы
    for idx in range(n_samples, n_rows * n_cols):
        i, j = idx // n_cols, idx % n_cols
        axes[i, j].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Unconditional generation inference plot saved: {save_path}")
    plt.show()
    plt.close(fig)


