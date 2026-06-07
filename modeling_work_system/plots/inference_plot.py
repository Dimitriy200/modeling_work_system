import matplotlib.pyplot as plt
import numpy as np


def plot_inference_results(
        y_true, 
        scenarios, 
        feature_idx=3,
        window_idx=0,
        feature_name="sensor measurement 2"):
    """
    Строит график сравнения истинных данных и сгенерированных сценариев VAE.
    
    Параметры:
    ----------
    y_true : np.ndarray
        Реальная матрица из 11 циклов формы (11, feature_dim)
    scenarios : list of np.ndarray
        Список сгенерированных сценариев от метода inference()
    feature_idx : int
        Индекс датчика (колонки), который хотим визуализировать
    feature_name : str
        Название датчика для заголовка графика
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Находим временные оси
    cycles = np.arange(1, 12) # 11 циклов суммарно (от 1 до 11)
    
    # 2. Рисуем истинные данные указанного sensor measurement (черная сплошная линия)
    plt.plot(cycles, y_true[:, feature_idx], color='black', linewidth=2.5, label='Реальные данные (NASA)')
    
    # 3. Рисуем сгенерированные сценарии (цветные полупрозрачные линии)
    # На отрезке 1-5 они будут показывать качество восстановления, на 6-11 - варианты будущего
    for i, scenario in enumerate(scenarios):
        # scenario имеет форму (1, 11, feature_dim), убираем размерность батча [0]
        gen_data = scenario[0, :, feature_idx]
        
        if i == 0:
            plt.plot(cycles, gen_data, color='crimson', alpha=0.4, linestyle='-', label='Сценарии VAE')
        else:
            plt.plot(cycles, gen_data, color='crimson', alpha=0.4, linestyle='-')
            
    # 4. Проводим вертикальную разделяющую линию после 5-го цикла
    plt.axvline(x=5, color='darkgray', linestyle='--', linewidth=2)

    # Общая инормация о графике
    plt.title(f"Валидация модели VAE | Окно данных №{window_idx} | {feature_name}", fontsize=14)

    # Подписи разделения на прошлое и будущее
    plt.text(3.0, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.9, 'История\n(Вход 1-5)', 
             horizontalalignment='center', color='gray', fontweight='bold')
    
    plt.text(8.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.9, 'Прогноз будущего\n(Генерация 6-11)', 
             horizontalalignment='center', color='gray', fontweight='bold')
    
    # Оформление графика
    plt.title(f"Генерация временной последовательности для: {feature_name}", fontsize=14)
    plt.xlabel("Номер цикла", fontsize=12)
    plt.ylabel("Нормализованное значение датчика", fontsize=12)
    plt.xticks(cycles)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    
    plt.show()
