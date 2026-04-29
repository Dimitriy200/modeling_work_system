
# ПОДБОР РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ

import keras
import logging
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve
)
from ..metrics.metrics import compute_mse
from ..models.basedetector import BaseAnomalyDetector


# ======================================================
# Поиск разделяющей поверхности стандартным образом
# ======================================================
def choose_optimal_threshold_stadart(
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


# ======================================================
# Поиск разделяющей поверхности новым образом
# ======================================================
def choose_optimal_threshold_un(
        model: BaseAnomalyDetector, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        feature_names: list = None,
        metric: str = 'f1',  # 'f1', 'precision', 'recall', 'balanced'
        target_recall: float = 0.95,  # для стратегии 'recall'
        plot: bool = True,
        run_id: str = None) -> dict:
    """
    Подбирает оптимальный порог реконструкционной ошибки (MSE) на валидационной выборке.
    
    Работает с данными из split_data_by_engine: X_val содержит и норму, и аномалию.
    
    Parameters
    ----------
    model : keras.Model
        Обученный автоэнкодер.
    X_val : pd.DataFrame
        Валидационные данные (признаки). Должны быть уже нормализованы.
    y_val : pd.Series
        Метки валидационных данных ('Norm'/'Anom' или 0/1).
    feature_names : list, optional
        Список колонок-сенсоров. Если None, берутся все числовые колонки.
    metric : str
        Стратегия выбора порога:
        - 'f1': максимизировать F1-score
        - 'precision': максимизировать Precision при Recall >= target_recall
        - 'recall': максимизировать Recall при Precision >= 0.5
        - 'balanced': точка, где Precision ≈ Recall
    target_recall : float
        Целевой уровень полноты для стратегий 'precision'/'recall'.
    plot : bool
        Построить график распределения ошибок и ROC-кривую.
    run_id : str, optional
        Идентификатор эксперимента для подписи графиков.
        
    Returns
    -------
    dict : {
        'threshold': float,          # выбранный порог
        'metrics': dict,             # все метрики на валидации
        'results_df': pd.DataFrame,  # детали по каждому образцу
        'plot_path': str or None     # путь к сохраненному графику
    }
    """
    
    logging.info(f"=== START CHOOSE THRESHOLD ===")

    # ======================================================
    # 1. Подготовка данных
    # ======================================================
    # Если передан DataFrame, берем только нужные фичи
    logging.info(f"Deleted non sensors columns...")
    if feature_names is None:
        # Автоматически исключаем не-сенсоры
        exclude = ['unit_number', 'cycle', 'label', 'is_anom', 'RUL']
        feature_names = [c for c in X_val.columns if c not in exclude and np.issubdtype(X_val[c].dtype, np.number)]
        logging.info(f"[Auto] Features selected: {len(feature_names)}")
    
    X_val_features = X_val[feature_names].values

    
    # Нормализация меток: приводим к бинарному виду (1 = норма, 0 = аномалия)
    # Поддерживаем разные форматы: 'Norm'/'Anom', 1/0, True/False
    # if y_val.dtype == object or y_val.dtype == str:
    #     y_val_binary = (y_val == self.split_info.get('normal_label', 'Norm')).astype(int)
    # else:
    #     # Если уже числа, предполагаем 1=норма, 0=аномалия (как в исходном коде)
    y_val_binary = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    
    logging.info(f"Validation: {len(X_val_features)} samples, Norm: {y_val_binary.sum()}, Anom: {(1-y_val_binary).sum()}")
    logging.info(f"Validation: {len(X_val_features)} samples, Norm: {y_val_binary}, Anom: {(1-y_val_binary)}")
    

    # ======================================================
    # 2. Предсказание и расчет ошибки 
    # ======================================================
    X_val_recon = model.predict(X_val_features, verbose=0)
    logging.info(f"X_val_features:\n{X_val_features}")
    logging.info(f"X_val_recon:\n{X_val_recon}")
    
    # Для алгоритма z1_core этот этам пропускаем
    # if X_val_features.shape == X_val_recon.shape and X_val_recon.shape == 2 and X_val_recon.shape == 2:
    #     mtk_errors = np.mean(np.square(X_val_features - X_val_recon), axis=1)
    # else:
    #     mtk_errors = X_val_recon
    mtk_errors = np.nanmax(np.square(X_val_features - X_val_recon), axis=1)
    
    logging.info(f"Reconstruction mse_errors:\n{mtk_errors}")

    # ======================================================
    # 3. Сбор результатов
    # ======================================================
    results_df = pd.DataFrame({
        'mse': mtk_errors,
        'true_class': y_val_binary,  # 1 = норма, 0 = аномалия
        'true_label': y_val.values if hasattr(y_val, 'values') else y_val  # оригинальная метка для отладки
    })
    
    
    # ======================================================
    # 4. Перебор порогов 
    # ======================================================
    # Оптимизация: берем не все уникальные MSE, а перцентили для скорости
    candidate_thresholds = np.percentile(mtk_errors, np.linspace(0, 100, 500))
    logging.info(f"Candidate thresholds:\n{candidate_thresholds}")

    
    best_threshold = None
    best_score = -1
    all_scores = []  # для графика
    
    for thr in candidate_thresholds:
        # Предсказание: MSE < порог → норма (1), иначе аномалия (0)
        pred_class = (mtk_errors < thr).astype(int)
        
        # Защита от деления на ноль и пустых предсказаний
        if pred_class.sum() == 0 or pred_class.sum() == len(pred_class):
            continue
            
        precision = precision_score(y_val_binary, pred_class, zero_division=0)
        recall = recall_score(y_val_binary, pred_class, zero_division=0)
        f1 = f1_score(y_val_binary, pred_class, zero_division=0)
        
        # Выбор стратегии
        if metric == 'f1':
            score = f1
        elif metric == 'precision':
            score = precision if recall >= target_recall else -1
        elif metric == 'recall':
            score = recall if precision >= 0.5 else -1
        elif metric == 'balanced':
            # Минимизируем разницу между Precision и Recall
            score = 1 - abs(precision - recall) if min(precision, recall) > 0 else -1
        else:
            score = f1  # fallback
        
        all_scores.append({'threshold': thr, 'precision': precision, 'recall': recall, 'f1': f1, 'score': score})
        
        if score > best_score:
            best_score = score
            best_threshold = thr
    
    if best_threshold is None:
        raise ValueError("Unable to find threshold. Please check your data and metrics..")
    

    # ======================================================
    # 5. Финальные метрики
    # ======================================================
    final_pred = (mtk_errors < best_threshold).astype(int)
    final_metrics = {
        'precision': precision_score(y_val_binary, final_pred, zero_division=0),
        'recall': recall_score(y_val_binary, final_pred, zero_division=0),
        'f1': f1_score(y_val_binary, final_pred, zero_division=0),
        'accuracy': (final_pred == y_val_binary).mean(),
        'roc_auc': roc_auc_score(y_val_binary, -mtk_errors),  # инвертируем, т.к. меньше MSE = лучше
        'threshold': best_threshold,
        'n_predictions': {
            'predicted_normal': int((final_pred == 1).sum()),
            'predicted_anomaly': int((final_pred == 0).sum()),
            'true_normal': int(y_val_binary.sum()),
            'true_anomaly': int((1 - y_val_binary).sum())
        }
    }
    
    results_df['pred_class'] = final_pred
    results_df['is_correct'] = (final_pred == y_val_binary).astype(int)
    
    logging.info(f"Threshold: {best_threshold:.6f}")
    logging.info(f"Metrics: F1={final_metrics['f1']:.4f}, Prec={final_metrics['precision']:.4f}, Rec={final_metrics['recall']:.4f}")
    
    # --- 6. Визуализация (опционально, для статьи) ---
    # plot_path = None
    # if plot:
    #     plot_path = self._plot_threshold_analysis(
    #         results_df, final_metrics, best_threshold,
    #         metric, run_id or 'threshold_analysis'
    #     )
    
    return {
        'threshold': float(best_threshold),
        'metrics': final_metrics,
        'results_df': results_df,
        'score_history': pd.DataFrame(all_scores)
        # 'plot_path': plot_path
    }