import numpy as np
import pandas as pd
import logging

from .aemetrics import AEMetricResult
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)


class ExperimentMetric:
    """
    Универсальный калькулятор метрик для экспериментов по детекции аномалий.
    
    Поддерживает:
    - Бинарную классификацию (норма/аномалия)
    - Ранжирующие метрики (ROC-AUC, PR-AUC)
    - Анализ матрицы ошибок
    - Экспорт в различные форматы
    """
    def __init__(
        self,
        normal_label: int = 1,
        anomaly_label: int = 0,
        zero_division: float = 0.0):
            
        self.normal_label = normal_label
        self.anomaly_label = anomaly_label
        self.zero_division = zero_division

        return None
    
# ======================================================
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        extra_metrics: Optional[Dict[str, float]] = None
    ) -> AEMetricResult:
        """
        Вычисляет все метрики эксперимента.
        
        Parameters
        ----------
        y_true : np.ndarray
            Истинные метки (1 = норма, 0 = аномалия).
        y_pred : np.ndarray
            Предсказанные метки (1 = норма, 0 = аномалия).
        scores : np.ndarray
            Скоры аномальности (непрерывные значения).
        threshold : float
            Порог классификации.
        extra_metrics : dict, optional
            Дополнительные метрики (например, из обучения модели).
            
        Returns
        -------
        MetricResult
            Объект со всеми метриками.
        """


        # Валидация входных данных
        # self._validate_inputs(y_true, y_pred, scores)
        
        # Базовые метрики
        precision = precision_score(
            y_true, y_pred, 
            pos_label=self.normal_label,
            zero_division=self.zero_division
        )
        recall = recall_score(
            y_true, y_pred, 
            pos_label=self.normal_label,
            zero_division=self.zero_division
        )
        f1 = f1_score(
            y_true, y_pred, 
            pos_label=self.normal_label,
            zero_division=self.zero_division
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # Ранжирующие метрики
        # Инвертируем скоры, т.к. больший скор = более аномально, а pos_label=1 (норма)
        roc_auc = roc_auc_score(y_true, -scores)
        pr_auc = average_precision_score(y_true, -scores)
        
        # Статистика предсказаний
        n_samples = len(y_true)
        n_true_normal = int(np.sum(y_true == self.normal_label))
        n_true_anomaly = int(np.sum(y_true == self.anomaly_label))
        n_pred_normal = int(np.sum(y_pred == self.normal_label))
        n_pred_anomaly = int(np.sum(y_pred == self.anomaly_label))
        
        # Матрица ошибок (детализация)
        # TN = правильно предсказанная норма, TP = правильно предсказанная аномалия
        # Но в sklearn: TN = [0,0], TP = [1,1] для labels=[0,1]
        cm = confusion_matrix(y_true, y_pred, labels=[self.normal_label, self.anomaly_label])
        
        # cm[0,0] = True Normal (норма → норма)
        # cm[0,1] = False Anomaly (норма → аномалия)
        # cm[1,0] = False Normal (аномалия → норма)
        # cm[1,1] = True Anomaly (аномалия → аномалия)
        n_true_positive = int(cm[0, 0])  # Правильно предсказанная норма
        n_true_negative = int(cm[1, 1])  # Правильно предсказанная аномалия
        n_false_positive = int(cm[0, 1])  # Ложная тревога
        n_false_negative = int(cm[1, 0])  # Пропущенная аномалия

        logging.info("Metrics calculation completed:")
        logging.info(f"precision = {precision}")
        logging.info(f"recall = {recall}")
        logging.info(f"f1 = {f1}")
        logging.info(f"accuracy = {accuracy}")
        logging.info(f"roc_auc = {roc_auc}")
        logging.info(f"pr_auc = {pr_auc}")
        logging.info(f"n_samples = {n_samples}")
        logging.info(f"n_true_normal = {n_true_normal}")
        logging.info(f"n_true_anomaly = {n_true_anomaly}")
        logging.info(f"n_pred_normal = {n_pred_normal}")
        logging.info(f"n_pred_anomaly = {n_pred_anomaly}")
        logging.info(f"n_true_positive = {n_true_positive}")
        logging.info(f"n_true_negative = {n_true_negative}")
        logging.info(f"n_false_positive = {n_false_positive}")
        logging.info(f"n_false_negative = {n_false_negative}")
        
        return AEMetricResult(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            roc_auc=float(roc_auc),
            pr_auc=float(pr_auc),
            n_samples=n_samples,
            n_true_normal=n_true_normal,
            n_true_anomaly=n_true_anomaly,
            n_pred_normal=n_pred_normal,
            n_pred_anomaly=n_pred_anomaly,
            n_true_positive=n_true_positive,
            n_true_negative=n_true_negative,
            n_false_positive=n_false_positive,
            n_false_negative=n_false_negative,
            threshold=float(threshold),
            extra_metrics=extra_metrics or {}
        )