# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging


import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_training_curves(history, save_path: str = None, figsize: tuple = (10, 6)):
    """
    Строит график потерь (loss) по эпохам.
    Автоматически обрабатывает как объект keras.History, так и словарь.
    """
    # 1. Извлекаем словарь метрик
    if hasattr(history, 'history'):
        metrics = history.history  # <-- ГЛАВНОЕ ИСПРАВЛЕНИЕ
    else:
        metrics = history
        
    # 2. Ищем правильный ключ для потерь (loss, train_loss, mse и т.д.)
    loss_key = 'loss'
    if 'loss' not in metrics:
        # Если точного совпадения нет, ищем первое слово с "loss"
        for key in metrics.keys():
            if 'loss' in key.lower():
                loss_key = key
                break
                
    # Проверка, что метрика вообще есть
    if loss_key not in metrics or not metrics[loss_key]:
        raise KeyError(f"Не найдены данные об обучении. Доступные метрики: {list(metrics.keys())}")
        
    epochs = range(1, len(metrics[loss_key]) + 1)
    
    # 3. Строим график
    plt.figure(figsize=figsize)
    plt.plot(epochs, metrics[loss_key], 'b-', label='Train Loss', linewidth=2)
    
    # Если есть валидационные потери, добавляем их
    if 'val_loss' in metrics:
        plt.plot(epochs, metrics['val_loss'], 'r--', label='Validation Loss', linewidth=2)
        
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 4. Сохранение
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ График сохранён: {save_path}")
    
    plt.show()