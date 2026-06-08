
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch

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



def plot_recursive_lifetime_forecast(
        model, 
        start_x_past, 
        full_engine_df, 
        feature_idx=11, 
        feature_name="sensor measurement 9", 
        num_scenarios=3, 
        save_path=None):
    """
    Генерирует автономный прогноз на всю жизнь двигателя, где каждое новое предсказание 
    строится на базе предыдущих сгенерированных моделью шагов.
    
    start_x_past: Тензор самого первого окна истории двигателя. Форма (1, 5, feature_dim)
    full_engine_df: Исходный pd.DataFrame ОДНОГО конкретного двигателя (все его ~175 строк)
    feature_idx: Индекс вашего датчика в отмасштабированном массиве (для Sensor 9 обычно 11 или 13)
    """
    model.eval()
    
    # 1. Извлекаем реальный тренд двигателя для подложки графика

    if hasattr(full_engine_df, 'values'):
        real_trend = full_engine_df.values[:, 2 + feature_idx]
    else:
        real_trend = full_engine_df[:, 2 + feature_idx]

    total_cycles = len(real_trend)
    cycles_axis = np.arange(1, total_cycles + 1)
    
    # Сюда будем складывать длинные сквозные траектории сценариев
    # Форма: (num_scenarios, total_cycles)
    all_scenarios_continuous = np.zeros((num_scenarios, total_cycles))
    
        # 2. Запускаем рекурсивный цикл генерации для каждого сценария отдельно
    with torch.no_grad():
        for s in range(num_scenarios):
            # Создаем копию стартовой истории (5, 26)
            current_history = start_x_past.clone()
            
            # Если случайно прилетел трехмерный тензор, сжимаем его до (5, 26)
            if current_history.dim() == 3:
                current_history = current_history.squeeze(0)
            
            # Записываем реальные первые 5 шагов в итоговый массив сценария
            all_scenarios_continuous[s, :5] = real_trend[:5]
            
            # Пошагово генерируем автономное будущее от 6-го цикла до самого конца жизни мотора
            for t in range(5, total_cycles):
                # Извлекаем последний шаг. Из матрицы (5, 26) строка [-1] имеет идеальную форму (26,)
                last_step = current_history[-1] 
                
                model_input_hist = current_history.unsqueeze(0)
                model_input_last = last_step.unsqueeze(0)
                
                # Модель предсказывает следующий шаг. На выходе получаем (1, 1, 26)
                y_next_pred, _, _ = model(model_input_hist, model_input_last)
                
                # Убираем размерности батча обратно, превращая шаг в вектор (26,)
                next_step_vector = y_next_pred.squeeze(0).squeeze(0) # Форма (26,)
                
                # Записываем сгенерированное значение нужного датчика в итоговый таймлайн
                all_scenarios_continuous[s, t] = next_step_vector[feature_idx].cpu().numpy()
                
                # ОБНОВЛЯЕМ ДВУМЕРНУЮ ИСТОРИЮ (Глубокая обратная связь):
                next_history_chunk = current_history[1:] # (4, 26)
                current_history = torch.cat([next_history_chunk, next_step_vector.unsqueeze(0)], dim=0) # Итог: снова (5, 26)!


                
    # 3. СТРОИМ ГРАФИК (Инвертируем цвета по вашему запросу!)
    plt.figure(figsize=(15, 6))
    
    # СИНЯЯ ЛИНИЯ - Истинные реальные значения (NASA)
    plt.plot(cycles_axis, real_trend, color='royalblue', linewidth=3.0, label='Реальные значения (NASA)')
    
    # СЕРЫЕ ЛИНИИ - Автономные рекурсивные генерации VAE
    for s in range(num_scenarios):
        if s == 0:
            plt.plot(cycles_axis, all_scenarios_continuous[s], color='darkgray', alpha=0.7, 
                     linewidth=1.5, linestyle='-', label='Рекурсивная генерация VAE')
        else:
            plt.plot(cycles_axis, all_scenarios_continuous[s], color='darkgray', alpha=0.7, linewidth=1.5, linestyle='-')
            
    # Вертикальная отметка окончания реальной предыстории
    plt.axvline(x=5, color='gray', linestyle='--', linewidth=1.5)
    
    plt.title(f"Долговременный автономный рекурсивный прогноз жизненного цикла | {feature_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Номер полетного цикла (Cycle)", fontsize=12)
    plt.ylabel("Значение датчика (Норм.)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
