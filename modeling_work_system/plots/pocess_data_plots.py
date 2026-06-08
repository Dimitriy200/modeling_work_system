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
