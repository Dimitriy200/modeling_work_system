
# ПОДБОР РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ


import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data.data_loader import load_csv_to_numpy
from evaluation.metrics import compute_mse


def choose_optimal_threshold(
    model,
    normal_control_path: str,
    anomaly_control_path: str,
    threshold_candidates: str = "all_mse_values"
) -> tuple[float, pd.DataFrame]:
    """
    Подбирает оптимальный порог реконструкционной ошибки (MSE) для разделения нормальных и аномальных данных.
    
    Args:
        model: Обученная модель Keras (автокодировщик).
        normal_control_path: Путь к CSV с нормальными данными (контрольная выборка).
        anomaly_control_path: Путь к CSV с аномальными данными (контрольная выборка).
        threshold_candidates: Стратегия выбора кандидатов. Сейчас поддерживается только "all_mse_values".
    
    Returns:
        tuple: (оптимальный_порог, DataFrame с полными результатами)
    """
    # Загрузка данных
    X_normal = load_csv_to_numpy(normal_control_path)
    X_anomaly = load_csv_to_numpy(anomaly_control_path)

    logging.info(f"Загружено нормальных данных: {X_normal.shape}, аномальных: {X_anomaly.shape}")

    # Предсказание реконструкции
    X_normal_recon = model.predict(X_normal, verbose=0)
    X_anomaly_recon = model.predict(X_anomaly, verbose=0)

    # Вычисление MSE
    mse_normal = compute_mse(X_normal, X_normal_recon)
    mse_anomaly = compute_mse(X_anomaly, X_anomaly_recon)

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

    return float(best_threshold), df_all