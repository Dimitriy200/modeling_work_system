"""
Модуль для инференса VAE-моделей и классификации аномалий.
"""

import torch
import numpy as np
import logging
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)


def calculate_reconstruction_error(
    model: torch.nn.Module,
    data_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """
    Вычисляет ошибку реконструкции (MSE) для каждого окна.
    
    Args:
        model: Обученная VAE модель.
        data_seq: Numpy array формы (N, seq_len, features).
        device: torch.device.
        batch_size: Размер батча для инференса.
    
    Returns:
        Numpy array ошибок реконструкции формы (N,).
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(data_seq), batch_size):
            batch = torch.tensor(
                data_seq[i:i+batch_size], 
                dtype=torch.float32
            ).to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(batch)
            
            # MSE по всем признакам и временным шагам для каждого семпла
            # shape: (batch_size, seq_len, features) -> (batch_size,)
            mse = torch.mean((batch - x_recon) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)


def classify_anomalies_by_percentile(
    model: torch.nn.Module,
    X_test_norm: np.ndarray,
    X_test_anom: np.ndarray,
    device: torch.device,
    percentile_threshold: float = 95.0,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Классифицирует данные на норму/аномалию по правилу процентиля.
    
    Args:
        model: Обученная VAE модель.
        X_test_norm: Тестовые нормальные данные (N, seq_len, features).
        X_test_anom: Тестовые аномальные данные (M, seq_len, features).
        device: torch.device.
        percentile_threshold: Процентиль для определения порога (по умолчанию 95).
        batch_size: Размер батча.
    
    Returns:
        Словарь с метриками, ошибками и предсказаниями.
    """
    logging.info("=" * 60)
    logging.info("ИНФЕРЕНС И КЛАССИФИКАЦИЯ АНОМАЛИЙ")
    logging.info("=" * 60)
    
    # 1. Расчет ошибок реконструкции
    logging.info(f"Расчет ошибок реконструкции для {len(X_test_norm)} нормальных окон...")
    errors_norm = calculate_reconstruction_error(model, X_test_norm, device, batch_size)
    
    logging.info(f"Расчет ошибок реконструкции для {len(X_test_anom)} аномальных окон...")
    errors_anom = calculate_reconstruction_error(model, X_test_anom, device, batch_size)
    
    # 2. Определение порога по процентилю нормальных данных
    threshold = float(np.percentile(errors_norm, percentile_threshold))
    logging.info(f"Порог ({percentile_threshold}-й процентиль нормы): {threshold:.6f}")
    
    # 3. Классификация
    # Норма: ошибка <= порога → предсказание 0 (норма)
    # Аномалия: ошибка > порога → предсказание 1 (аномалия)
    y_pred_norm = (errors_norm > threshold).astype(int)
    y_pred_anom = (errors_anom > threshold).astype(int)
    
    # Истинные метки
    y_true_norm = np.zeros_like(errors_norm)
    y_true_anom = np.ones_like(errors_anom)
    
    # Объединяем для расчета общих метрик
    y_true = np.concatenate([y_true_norm, y_true_anom])
    y_pred = np.concatenate([y_pred_norm, y_pred_anom])
    y_scores = np.concatenate([errors_norm, errors_anom])
    
    # 4. Расчет метрик
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'threshold': threshold,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
    }
    
    # Логирование результатов
    logging.info("\n=== МЕТРИКИ КЛАССИФИКАЦИИ ===")
    logging.info(f"Accuracy:   {acc:.4f}")
    logging.info(f"Precision:  {prec:.4f}")
    logging.info(f"Recall:     {rec:.4f}")
    logging.info(f"F1-Score:   {f1:.4f}")
    logging.info(f"ROC-AUC:    {roc_auc:.4f}")
    logging.info(f"Specificity:{metrics['specificity']:.4f}")
    logging.info(f"\nПорог: {threshold:.6f}")
    logging.info(f"\nConfusion Matrix:")
    logging.info(f"  TP (True Positives):  {tp} (правильно обнаружены аномалии)")
    logging.info(f"  TN (True Negatives):  {tn} (правильно обнаружена норма)")
    logging.info(f"  FP (False Positives): {fp} (ложные срабатывания)")
    logging.info(f"  FN (False Negatives): {fn} (пропущенные аномалии)")
    
    # Статистика ошибок
    logging.info(f"\n=== СТАТИСТИКА ОШИБОК РЕКОНСТРУКЦИИ ===")
    logging.info(f"Норма:    mean={errors_norm.mean():.6f}, std={errors_norm.std():.6f}")
    logging.info(f"Аномалия: mean={errors_anom.mean():.6f}, std={errors_anom.std():.6f}")
    logging.info(f"Gap (аномалия - норма): {errors_anom.mean() - errors_norm.mean():.6f}")
    
    return {
        'metrics': metrics,
        'errors_norm': errors_norm,
        'errors_anom': errors_anom,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores,
        'threshold': threshold
    }


import matplotlib.pyplot as plt
import seaborn as sns


def plot_classification_results(
    inference_results: Dict[str, Any],
    save_path: str,
    figsize: Tuple[int, int] = (18, 5)
) -> None:
    """
    Строит комплексный график результатов классификации.
    
    Args:
        inference_results: Результат функции classify_anomalies_by_percentile.
        save_path: Путь для сохранения PNG.
        figsize: Размер фигуры.
    """
    errors_norm = inference_results['errors_norm']
    errors_anom = inference_results['errors_anom']
    threshold = inference_results['threshold']
    metrics = inference_results['metrics']
    y_true = inference_results['y_true']
    y_scores = inference_results['y_scores']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        'Anomaly Detection via Reconstruction Error\n'
        f'Threshold: {threshold:.4f} | ROC-AUC: {metrics["roc_auc"]:.3f} | '
        f'F1: {metrics["f1_score"]:.3f}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    # === График 1: Распределение ошибок ===
    ax1 = axes[0]
    sns.histplot(
        errors_norm, color='green', label='Normal (Test)', 
        alpha=0.6, bins=50, ax=ax1, kde=True
    )
    sns.histplot(
        errors_anom, color='red', label='Anomaly (Test)', 
        alpha=0.6, bins=50, ax=ax1, kde=True
    )
    ax1.axvline(
        threshold, color='black', linestyle='--', linewidth=2,
        label=f'Threshold ({threshold:.3f})'
    )
    ax1.set_title('Reconstruction Error Distribution')
    ax1.set_xlabel('MSE Error')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === График 2: ROC-кривая ===
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = metrics['roc_auc']
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === График 3: Confusion Matrix ===
    ax3 = axes[2]
    cm = confusion_matrix(y_true, inference_results['y_pred'])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly']
    )
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    ax3.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Classification results plot saved: {save_path}")
    plt.show()
    plt.close(fig)