# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging


def plot_training_curves(
        history: dict, 
        save_path: str = None, 
        figsize: tuple = (10, 6)):
    """
    Строит график train_loss и val_loss по эпохам.
    
    Parameters
    ----------
    history : dict
        history.history от model.fit() — {'loss': [...], 'val_loss': [...]}
    save_path : str, optional
        Если указан — сохраняет график в файл.
    """
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"График сохранён: {save_path}")
    
    plt.show()