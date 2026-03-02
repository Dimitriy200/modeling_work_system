
# ПОДБОР РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ

import keras
import logging
import numpy as np
import pandas as pd
import mlflow

from sklearn.metrics import accuracy_score
from .metrics import compute_mse


def choose_optimal_threshold(
    model: keras.Model,
    normal_control_df: np.ndarray,
    anomaly_control_df: np.ndarray,
    run_id: str = None,
    threshold_candidates: str = "all_mse_values" ) -> tuple[float, pd.DataFrame]:
    """
    Подбирает оптимальный порог реконструкционной ошибки (MSE) для разделения нормальных и аномальных данных.
    
    Args:
        model: Обученная модель Keras (автокодировщик).
        normal_control_path: Путь к CSV c нормальными данными (контрольная выборка).
        anomaly_control_path: Путь к CSV c аномальными данными (контрольная выборка).
        threshold_candidates: Стратегия выбора кандидатов. Сейчас поддерживается только "all_mse_values".
    
    Returns:
        tuple: (oптимaльный_пopoг, DataFrame c полными результатами)
    """
    
    logging.info(f"Загружено нормальных данных: {normal_control_df.shape}, аномальных: {anomaly_control_df.shape}")

    # Предсказание реконструкции
    X_normal_recon = model.predict(normal_control_df, verbose=0)
    X_anomaly_recon = model.predict(anomaly_control_df, verbose=0)

    # Вычисление MSE
    mse_normal = compute_mse(normal_control_df, X_normal_recon)
    mse_anomaly = compute_mse(anomaly_control_df, X_anomaly_recon)

    # Создание DataFrame
    df_normal = pd.DataFrame({
        "mse": mse_normal,
        "true_class": 1  # норма = 1
    })

    df_anomaly = pd.DataFrame({
        "mse": mse_anomaly,
        "true_class": 0  # аномалия = 0
    })

    df_all = pd.concat([df_normal, df_anomaly], ignore_index=True)
    logging.info(f"Объединённый датасет: {df_all.shape}")

    # Кандидаты на порог — все уникальные значения MSE (отсортированы)
    candidate_thresholds = np.sort(df_all["mse"].unique())

    best_threshold = 0.0
    best_accuracy = -1.0
    best_predictions = None

    # Перебор всех возможных порогов
    for thr in candidate_thresholds:
        # Предсказание: если MSE < порог → норма (1), иначе аномалия (0)
        pred_class = (df_all["mse"] < thr).astype(int)
        acc = accuracy_score(df_all["true_class"], pred_class)

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thr
            best_predictions = pred_class

    # Сохраняем финальные предсказания
    df_all["pred_class"] = best_predictions
    logging.info(f"Оптимальный порог: {best_threshold:.6f}, точность: {best_accuracy:.4f}")

    return float(best_threshold), float(best_accuracy),  df_all