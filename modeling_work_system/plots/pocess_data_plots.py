import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging



def plot_sensor_smoothing(
        df_raw, 
        df_smoothed, 
        engine_id=1, 
        feature_name="s_2", 
        save_path=None,
        col_group="unit number"):
    """
    Строит график сравнения показаний датчика до и после сглаживания для конкретного двигателя.
    
    df_raw: Исходный зашумленный pd.DataFrame от NASA
    df_smoothed: Сглаженный pd.DataFrame (после применения rolling или ewm)
    engine_id: ID двигателя (unit_number), который хотим посмотреть (например, 1)
    feature_name: Название колонки датчика (например, 's_2' или 'sensor_measurement_2')
    """
    # 1. Фильтруем данные строго для одного выбранного двигателя
    engine_raw = df_raw[df_raw[col_group] == engine_id]
    engine_smooth = df_smoothed[df_smoothed[col_group] == engine_id]
    
    # Извлекаем ось X (номера циклов) и оси Y (показания датчиков)
    cycles_raw = engine_raw['time in cycles'].values
    y_raw = engine_raw[feature_name].values
    y_smooth = engine_smooth[feature_name].values
    
    # 2. СТРОИМ ГРАФИК
    plt.figure(figsize=(14, 6))
    
    # Рисуем исходный шумный датчик (тонкая серая линия со светлыми точками)
    plt.plot(cycles_raw, y_raw, color='darkgray', alpha=0.6, linewidth=1.5, 
             label='Исходные данные NASA (С датчиков + Шум)')
    plt.scatter(cycles_raw, y_raw, color='gray', alpha=0.4, s=15)
    
    # Рисуем сглаженный тренд (толстая синяя или зеленая линия)
    plt.plot(cycles_raw, y_smooth, color='royalblue', linewidth=3.0, 
             label='Сглаженный тренд (Фильтр Rolling Mean)')
    
    # Оформление графика
    plt.title(f"Сравнение сигнала до и после сглаживания | Двигатель №{engine_id} | {feature_name}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Номер полетного цикла (Cycle)", fontsize=12)
    plt.ylabel("Значение датчика (Физическая величина)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    # Сначала сохраняем, потом выводим на экран, чтобы избежать пустого холста
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"The smoothing graph has been saved successfully.: {save_path}")
        
    plt.show()


def plot_sensor_stress_testing(
        array_clean, 
        array_stressed, 
        group_number=1, 
        feature_idx=6, 
        feature_name="sensor measurement 2", 
        save_path=None):
    """
    Отрисовывает графиг зашумления данных
    """
    if hasattr(array_clean, 'values'): array_clean = array_clean.values
    if hasattr(array_stressed, 'values'): array_stressed = array_stressed.values

    window_idx = group_number - 1

    if array_clean.ndim == 3:
        y_clean = array_clean[window_idx, :, feature_idx]
        y_stressed = array_stressed[window_idx, :, feature_idx]
    else:
        # === Единая маска по чистой матрице ===
        unique_engine_ids = np.unique(array_clean[:, 0])
        target_normalized_id = unique_engine_ids[window_idx]
        
        # Строим маску строго по ЧИСТОМУ массиву, где ID точно целы!
        mask = array_clean[:, 0] == target_normalized_id
        
        # Вырезаем одинаковые строки из обеих матриц по чистой маске
        engine_clean_data = array_clean[mask]
        engine_stressed_data = array_stressed[mask]
        
        y_clean = engine_clean_data[:, feature_idx]
        y_stressed = engine_stressed_data[:, feature_idx]
        
        logging.info(f"[STRESS_PLOT] Синхронное извлечение 2D. Длина тренда: {len(y_clean)}")

    # Автоматическое выравнивание на случай непредвиденных сдвигов
    min_len = min(len(y_clean), len(y_stressed))
    if min_len == 0:
        logging.error("[STRESS_PLOT] Ошибка: Извлеченные массивы пусты!")
        return
        
    y_clean = y_clean[:min_len]
    y_stressed = y_stressed[:min_len]
    cycles = np.arange(1, min_len + 1)

    plt.figure(figsize=(15, 6))
    plt.plot(
        cycles, 
        y_clean, 
        color='royalblue', 
        linewidth=2.0,
        label='Чистый сигнал', 
        zorder=2)
        
    plt.scatter(
        cycles, 
        y_clean, 
        color='royalblue', 
        # alpha=0.5,
        s=10, 
        zorder=2)
    
    plt.plot(
        cycles, 
        y_stressed, 
        color='darkgray', 
        linewidth=1.2, 
        linestyle='--', 
        alpha=0.8,
        label='Зашумленный сигнал', 
        zorder=3)
    
    zero_mask = y_stressed == 0.0
    if np.any(zero_mask):
        plt.scatter(cycles[zero_mask], y_stressed[zero_mask], color='black', marker='x', s=30, linewidths=2.0, label='Пропуск (0.0)', zorder=4)
    plt.scatter(cycles[~zero_mask], y_stressed[~zero_mask], color='darkgray', alpha=0.5, s=10, zorder=3)

    plt.title(f"Визуализация искуственного зашумленя сигнала | {feature_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Порядковый номер цикла", fontsize=12)
    plt.ylabel("Нормализованное значение", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
