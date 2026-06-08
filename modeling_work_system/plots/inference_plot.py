import matplotlib.pyplot as plt
import numpy as np
import torch




def plot_inference_results(
        y_true, 
        scenarios,
        save_path: str,
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
    total_cycles = y_true.shape[0] 
    cycles = np.arange(1, total_cycles+1) # 10 циклов суммарно (от 1 до 11)
    
    # 2. Рисуем истинные данные указанного sensor measurement (черная сплошная линия)
    plt.plot(cycles, y_true[:, feature_idx], color='black', marker='o', linewidth=2.5, label='Реальные данные (NASA)')
    
    # 3. Рисуем сгенерированные сценарии (цветные полупрозрачные линии)
    # На отрезке 1-5 они будут показывать качество восстановления, на 6-11 - варианты будущего
    for i, scenario in enumerate(scenarios):
        # scenario имеет форму (1, 11, feature_dim), убираем размерность батча [0]
        gen_data = scenario[:, feature_idx]
        
        if i == 0:
            plt.plot(cycles, gen_data, color='crimson', alpha=0.4, linestyle='-', marker='o', label='Сценарии VAE')
        else:
            plt.plot(cycles, gen_data, color='crimson', alpha=0.4, linestyle='-', marker='o')
            
    # 4. Проводим вертикальную разделяющую линию после 5-го цикла
    plt.axvline(x=5, color='darkgray', linestyle='--', linewidth=2)

    # Общая инормация о графике
    plt.title(f"Валидация модели VAE | Окно данных №{window_idx} | {feature_name}", fontsize=14)

    # Подписи разделения на прошлое и будущее
    plt.text(3.0, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.9, 'История\n', 
             horizontalalignment='center', color='gray', fontweight='bold')
    
    plt.text(8.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.9, 'Прогноз будущего\n', 
             horizontalalignment='center', color='gray', fontweight='bold')
    
    # Оформление графика
    plt.title(f"Генерация временной последовательности для: {feature_name}", fontsize=14)
    plt.xlabel("Номер цикла", fontsize=12)
    plt.ylabel("Нормализованное значение датчика", fontsize=12)
    plt.xticks(cycles)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_inference_multi_features(
        y_true, 
        scenarios, 
        feature_indices=[2, 9], 
        feature_names=["Sensor 2", "Sensor 9"], save_path=None):
    """
    Строит графики для нескольких датчиков одного двигателя на одном холсте (друг под другом).
    
    Параметры:
    ----------
    y_true : np.ndarray
        Реальная матрица одного двигателя формы (10, feature_dim)
    scenarios : list of np.ndarray
        Список сгенерированных сценариев для одного двигателя, где каждый формы (10, feature_dim)
    feature_indices : list of int
        Список индексов датчиков, которые нужно отобразить (например, [2, 9])
    feature_names : list of str
        Названия датчиков для заголовков панелей
    """
    num_features = len(feature_indices)
    total_cycles = y_true.shape[0] 
    cycles = np.arange(1, total_cycles + 1) 
    
    # Создаем холст: num_features строк, 1 колонка. Автоматически масштабируем высоту.
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features), sharex=True)
    
    # Если датчик только один, matplotlib возвращает не список осей, а один объект. Делаем его списком.
    if num_features == 1:
        axes = [axes]
        
    for idx, (f_idx, f_name) in enumerate(zip(feature_indices, feature_names)):
        ax = axes[idx]
        
        # 1. Рисуем реальные данные для текущего датчика (черная линия)
        ax.plot(cycles, y_true[:, f_idx], color='black', marker='o', linewidth=2.5, label='Реальные данные (NASA)')
        
        # 2. Рисуем веер альтернативных сценариев VAE (розовые линии)
        for i, scenario in enumerate(scenarios):
            gen_data = scenario[:, f_idx]
            if i == 0:
                ax.plot(cycles, gen_data, color='crimson', marker='o', alpha=0.4, linestyle='-', label='Сценарии VAE')
            else:
                ax.plot(cycles, gen_data, color='crimson', marker='o', alpha=0.4, linestyle='-')
                
        # 3. Вертикальная линия разделения после 5-го шага известной истории
        ax.axvline(x=5, color='darkgray', linestyle='--', linewidth=2)
        
        # Разметка подписей "История" / "Прогноз" только для верхнего графика, чтобы не дублировать
        if idx == 0:
            ax.text(3.0, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9, 'История (1-5)', 
                    horizontalalignment='center', color='gray', fontweight='bold')
            ax.text(8.5, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9, 'Прогноз (6-10)', 
                    horizontalalignment='center', color='gray', fontweight='bold')
            
        # Настройка оформления для каждой панели
        ax.set_title(f"Генерация для: {f_name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Нормализ. значение", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left')
        
    # Общая подпись для оси X в самом низу
    plt.xlabel("Номер цикла", fontsize=12)
    plt.xticks(cycles)
    
    plt.tight_layout()
    
    # Сохраняем перед plt.show(), чтобы не получить пустой холст
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Мульти-график успешно сохранен: {save_path}")
        
    plt.show()




def plot_continuous_forecasting(model, X_val_past, X_val_ls, y_val_true, num_scenarios=5, feature_idx=6, feature_name="Sensor 2", save_path=None):
    """
    Собирает предсказания из всех последовательных окон в одну длинную непрерывную линию.
    
    y_val_true: массив ВСЕХ полных окон валидации формы (N, 10, feature_dim)
    """
    model.eval()
    num_windows = len(X_val_past)
    
    # 1. Собираем ИСТИННЫЙ непрерывный тренд
    # Берем 1-5 шаги из самого первого окна, а затем строго 6-й шаг из каждого последующего окна
    true_continuous = list(y_val_true[0, :5, feature_idx])
    for i in range(num_windows):
        true_continuous.append(y_val_true[i, 5, feature_idx]) # 6-й шаг (индекс 5)
        
    true_continuous = np.array(true_continuous)
    total_length = len(true_continuous)
    cycles = np.arange(1, total_length + 1)
    
    # 2. Инициализируем массивы для сценариев генерации VAE
    # Форма: (num_scenarios, total_length)
    scenarios_continuous = np.zeros((num_scenarios, total_length))
    
    # Заполняем первые 5 шагов каждого сценария истинной предысторией (так как это зона восстановления)
    for s in range(num_scenarios):
        scenarios_continuous[s, :5] = y_val_true[0, :5, feature_idx]
        
    # 3. Запускаем инференс по всем окнам последовательно
    with torch.no_grad():
        # Чтобы не перегружать память, делаем инференс батчем для всех окон сразу
        # Получаем список из num_scenarios, каждый формы (N, 10, feature_dim)
        gen_scenarios = model.inference(
            x_past=torch.FloatTensor(X_val_past),
            horizon=10,
            num_scenarios=num_scenarios
        )
        
        # Вытаскиваем строго 6-й шаг (индекс 5) из каждого окна для каждого сценария
        for s in range(num_scenarios):
            for i in range(num_windows):
                # Записываем предсказание в общую временную шкалу (со сдвигом на 5 шагов предыстории)
                scenarios_continuous[s, 5 + i] = gen_scenarios[s][i, 5, feature_idx]

    # 4. СТРОИМ ГРАФИК
    plt.figure(figsize=(15, 6))
    
    # Рисуем реальный длинный тренд датчика
    plt.plot(cycles, true_continuous, color='black', linewidth=2.5, label='Реальный непрерывный тренд (NASA)')
    
    # Рисуем длинные непрерывные сценарии VAE
    for s in range(num_scenarios):
        if s == 0:
            plt.plot(cycles, scenarios_continuous[s], color='crimson', alpha=0.4, label='Непрерывный прогноз VAE')
        else:
            plt.plot(cycles, scenarios_continuous[s], color='crimson', alpha=0.4)
            
    # Вертикальная линия, отделяющая самую первую известную историю от зоны глобального прогноза
    plt.axvline(x=5, color='darkgray', linestyle='--', linewidth=2)
    
    plt.title(f"Глобальный непрерывный прогноз по всей цепочке окон: {feature_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Абсолютный номер цикла работы двигателя", fontsize=12)
    plt.ylabel("Значение датчика (Норм.)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_consecutive_windows(model, X_val_past, X_val_seq_full, start_window_idx=0, num_windows=5, feature_idx=6, feature_name="Sensor 2", save_path=None):
    """
    Берет num_windows (например, 5) окон подряд для одного двигателя,
    склеивает их в короткий читаемый непрерывный тренд и выводит прогноз.
    """
    model.eval()
    
    # 1. Вырезаем блок из 5 последовательных окон
    end_window_idx = start_window_idx + num_windows
    X_past_block = X_val_past[start_window_idx:end_window_idx]
    X_full_block = X_val_seq_full[start_window_idx:end_window_idx]
    
    # 2. Собираем ИСТИННЫЙ непрерывный тренд для этого короткого участка
    # Берем первые 5 шагов из самого первого окна блока
    true_trend = list(X_full_block[0, :5, feature_idx])
    # А затем строго 6-й шаг (индекс 5) из каждого из 5 окон
    for i in range(num_windows):
        true_trend.append(X_full_block[i, 5, feature_idx])
        
    true_trend = np.array(true_trend)
    total_len = len(true_trend) # Для 5 окон длина будет ровно 5 + 5 = 10 шагов глобального тренда
    cycles = np.arange(1, total_len + 1)
    
    # 3. Запускаем инференс VRNN для выбранного блока окон
    with torch.no_grad():
        # Получаем список из сценариев, каждый формы (num_windows, 10, feature_dim)
        gen_scenarios = model.inference(
            x_past=torch.FloatTensor(X_past_block),
            horizon=10,
            num_scenarios=5
        )
    
    # 4. СТРОИМ ГРАФИК
    plt.figure(figsize=(12, 6))
    
    # Рисуем истинный тренд датчика (черная линия)
    plt.plot(cycles, true_trend, color='black', linewidth=3, label='Реальный тренд (NASA)', zorder=3)
    
    # Рисуем предсказания. Для каждого окна мы нарисуем его индивидуальный "веер" будущего (шаги 6-10)
    # Чтобы график оставался читаемым, мы будем плавно смещать каждый веер вправо
    for s in range(len(gen_scenarios)): # По всем сценариям (5 штук)
        scenario_data = gen_scenarios[s] # Массив формы (num_windows, 10, feature_dim)
        
        for w in range(num_windows):
            # Будущее для окна w начинается после 5 шагов его собственной истории
            # На глобальной шкале X это соответствует точкам от (w + 1) до (w + 6)
            x_scen_cycles = np.arange(w + 1, w + 11)
            gen_data_window = scenario_data[w, :, feature_idx]
            
            if s == 0 and w == 0:
                plt.plot(x_scen_cycles, gen_data_window, color='crimson', alpha=0.25, linestyle='-', label='Прогнозы VRNN (5 окон)')
            else:
                plt.plot(x_scen_cycles, gen_data_window, color='crimson', alpha=0.15, linestyle='-')
                
    # Визуальные разделители: проведем вертикальные пунктиры для каждого окна, чтобы видеть их старты
    for w in range(num_windows):
        plt.axvline(x=w+5, color='gray', linestyle=':', alpha=0.4)
        plt.text(w+5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.02, f"Окно {w+1}", 
                 horizontalalignment='center', color='gray', fontsize=8)

    plt.title(f"Микро-анализ последовательных окон (сдвиг STRIDE=1) | {feature_name}", fontsize=13, fontweight='bold')
    plt.xlabel("Относительные циклы времени на выбранном участке", fontsize=11)
    plt.ylabel("Значение датчика (Норм.)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()