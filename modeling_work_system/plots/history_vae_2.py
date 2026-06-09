import matplotlib.pyplot as plt

def plot_vae_training_history(history, save_path: str):
    # Создаем окно с двумя графиками рядом
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # --- ПАНЕЛЬ 1: МЕТРИКИ ТОЧНОСТИ ПРОГНОЗА ---
    ax1.plot(history['total_loss'], label='Общий Loss (Total Loss)', color='forestgreen', linestyle='--', zorder=3)
    ax1.plot(history['mse_loss'], label='Ошибка прогноза (MSE Loss)', color='royalblue', linewidth=2, zorder=2)
    ax1.set_title('Точность восстановления и прогноза датчиков', fontsize=12)
    ax1.set_xlabel('Эпохи обучения', fontsize=10)
    ax1.set_ylabel('Значение ошибки', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # --- ПАНЕЛЬ 2: МЕТРИКИ ЛАТЕНТНОГО ПРОСТРАНСТВА ---
    ax2.plot(history['kl_loss'], label='Латентный хаос (KLD Loss)', color='darkorange', linewidth=2)
    ax2.set_title('Состояние скрытого пространства (KLD)', fontsize=12)
    ax2.set_xlabel('Эпохи обучения', fontsize=10)
    ax2.set_ylabel('Значение KLD', color='darkorange', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Создаем вторую шкалу Y для отображения графика включения веса KL
    ax3 = ax2.twinx()
    ax3.plot(history['kl_weight'], label='График включения KL (Weight)', color='forestgreen', linestyle=':', linewidth=2)
    ax3.set_ylabel('Вес коэффициента KL (от 0 до 1)', color='forestgreen', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='forestgreen')

    # Объединяем легенды для правого графика в одну коробку
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(handles2 + handles3, labels2 + labels3, loc='upper left')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
