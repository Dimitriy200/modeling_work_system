import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

def calculate_reconstruction_errors(
    model: torch.nn.Module, 
    data_seq: np.ndarray, 
    device: torch.device,
    chunk_size: int = 64
) -> np.ndarray:
    """
    Вычисляет ошибку реконструкции (MSE) для каждого семпла в последовательности.
    
    Args:
        model: Обученная модель VAE.
        data_seq: Numpy массив формы (N_samples, seq_len, n_features).
        device: 'cuda' или 'cpu'.
        chunk_size: Размер батча для инференса (чтобы не переполнить VRAM).
        
    Returns:
        Numpy массив ошибок реконструкции формы (N_samples,).
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(data_seq), chunk_size):
            batch = torch.tensor(data_seq[i:i+chunk_size], dtype=torch.float32).to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(batch)
            
            # MSE по всем признакам и временным шагам для каждого семпла в батче
            # shape: (batch_size, seq_len, n_features) -> (batch_size,)
            mse = torch.mean((batch - x_recon) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())
            
    return np.array(errors)


def evaluate_and_plot_vae(
    model: torch.nn.Module,
    X_test_norm: np.ndarray,
    X_test_anom: np.ndarray,
    device: torch.device,
    save_path: str,
    percentile_threshold: float = 95.0,
    feature_idx_to_plot: int = 1
) -> Dict[str, float]:
    """
    Оценивает качество модели VAE, классифицирует данные по правилу процентиля 
    и строит комплексный график оценки.
    
    Args:
        model: Обученная модель VAE.
        X_test_norm: Тестовые нормальные данные (N, seq_len, features).
        X_test_anom: Тестовые аномальные данные (N, seq_len, features).
        device: Устройство для вычислений.
        save_path: Путь для сохранения итогового изображения.
        percentile_threshold: Процентиль для определения порога аномалии (по умолчанию 95).
        feature_idx_to_plot: Индекс признака для графика реконструкции.
        
    Returns:
        Словарь с метриками классификации и значением порога.
    """
    logging.info("Calculating reconstruction errors for Test Normal data...")
    errors_norm = calculate_reconstruction_errors(model, X_test_norm, device)
    
    logging.info("Calculating reconstruction errors for Test Anomaly data...")
    errors_anom = calculate_reconstruction_errors(model, X_test_anom, device)
    
    # 1. Определение порога по заданному процентилю нормальных данных
    threshold = float(np.percentile(errors_norm, percentile_threshold))
    logging.info(f"{percentile_threshold}th percentile threshold (Normal data): {threshold:.4f}")
    
    # 2. Классификация: ошибка > порога = Аномалия (1), иначе Норма (0)
    y_true_norm = np.zeros_like(errors_norm)
    y_pred_norm = (errors_norm > threshold).astype(int)
    
    y_true_anom = np.ones_like(errors_anom)
    y_pred_anom = (errors_anom > threshold).astype(int)
    
    # Объединяем для расчета общих метрик
    y_true = np.concatenate([y_true_norm, y_true_anom])
    y_pred = np.concatenate([y_pred_norm, y_pred_anom])
    y_scores = np.concatenate([errors_norm, errors_anom]) # Используем саму ошибку как score для ROC
    
    # 3. Расчет метрик
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    metrics['roc_auc'] = auc(fpr, tpr)
    
    logging.info("=== CLASSIFICATION METRICS ===")
    logging.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall:    {metrics['recall']:.4f}")
    logging.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logging.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # 4. Визуализация
    plt.figure(figsize=(18, 5))
    
    # График 1: Распределение ошибок реконструкции
    plt.subplot(1, 3, 1)
    sns.histplot(errors_norm, color='green', label='Normal (Test)', alpha=0.6, bins=50)
    sns.histplot(errors_anom, color='red', label='Anomaly (Test)', alpha=0.6, bins=50)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, 
                label=f'{percentile_threshold}th Percentile\n({threshold:.2f})')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('MSE Error')
    plt.ylabel('Frequency')
    plt.legend()
    
    # График 2: ROC Кривая
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # График 3: Пример реконструкции (берем первый аномальный семпл, если он есть)
    plt.subplot(1, 3, 3)
    if len(X_test_anom) > 0:
        sample_idx = 0
        x_orig = X_test_anom[sample_idx]
        
        with torch.no_grad():
            x_tensor = torch.tensor([x_orig], dtype=torch.float32).to(device)
            x_rec, _, _ = model(x_tensor)
            x_rec = x_rec.cpu().numpy()[0]
        
        plt.plot(x_orig[:, feature_idx_to_plot], label='Original (Anomaly)', marker='o', markersize=3)
        plt.plot(x_rec[:, feature_idx_to_plot], label='Reconstructed', marker='x', markersize=3, linestyle='--')
        plt.title(f'Reconstruction of Feature {feature_idx_to_plot}\n(Anomaly Sample)')
        plt.xlabel('Time Step (in window)')
        plt.ylabel('Scaled Value')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No anomaly samples available', ha='center', va='center')
        plt.title('Reconstruction Example')
    
    plt.tight_layout()
    
    # Сохранение и отображение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Evaluation plots saved to: {save_path}")
    plt.show()
    plt.close()
    
    return metrics