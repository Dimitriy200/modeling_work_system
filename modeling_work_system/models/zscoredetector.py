# src/models/zscore_detector.py
import numpy as np
import logging

from typing import Optional, Union
from .basedetector import BaseAnomalyDetector
import mlflow
import numpy as np
import pandas as pd
import tempfile
import os
import json
import joblib
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, accuracy_score, confusion_matrix
)


class ZScoreDetector(BaseAnomalyDetector):
    """
    Статистический детектор: аномалия = отклонение > k*std от среднего.
    """
    
    def __init__(self, k: float = 3.0, aggregation: str = 'max'):
        """
        Parameters
        ----------
        k : float
            Порог в единицах стандартного отклонения.
        aggregation : str
            Как агрегировать Z-score по всем признакам: 'max', 'mean', 'l2'.
        """
        self.k = k
        self.aggregation = aggregation
        self.mean_ = None
        self.std_ = None
        
# ======================================================
    def fit(self, 
            X_train: np.ndarray, 
            y_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            verbose=None) -> 'ZScoreDetector':
        
        X_array = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else np.asarray(X_train)
        self.mean_ = np.mean(X_array, axis=0)  # Форма: (26,)
        self.std_ = np.std(X_array, axis=0)    # Форма: (26,)
        
        return self
    
# ======================================================
    def predict(self, X: np.ndarray, verbose=None) -> np.ndarray:
        
        # X_array = X.values if hasattr(X, 'values') else X
        # z_scores = np.abs((X - self.mean_) / self.std_)

        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        # Вычисляем z-scores для КАЖДОГО признака
        z_scores = np.abs((X_array - self.mean_) / self.std_)
        logging.info(f"Z scores: \n{z_scores}")
        
        if self.aggregation == 'max':
            res = np.nanmax(z_scores, axis=1)
            logging.info(f"Z scores max result: \n{res}")
            return res
        elif self.aggregation == 'mean':
            return np.mean(z_scores, axis=1)
        elif self.aggregation == 'l2':
            return np.sqrt(np.sum(z_scores**2, axis=1))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

# ======================================================
    def choose_optimal_threshold_universal(
        model,
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        split_info: dict,
        feature_names: list = None,
        metric: str = 'f1',
        target_recall: float = 0.95,
        use_reconstruction: bool = None  # None = автоопределение
    ) -> dict:
        """
        Универсальный подбор порога для любых моделей детекции аномалий.
        
        Поддерживает:
        - Автоэнкодеры: скор = MSE реконструкции
        - Статистические модели: скор = output predict_scores()
        
        Parameters
        ----------
        model : BaseAnomalyDetector
            Обученная модель.
        X_val : array-like
            Валидационные признаки.
        y_val : array-like
            Метки валидации.
        split_info : dict
            Информация о сплите (нужна для normal_label).
        feature_names : list, optional
            Имена признаков.
        metric : str
            Метрика для оптимизации: 'f1', 'precision', 'recall', 'balanced'.
        target_recall : float
            Целевой Recall для стратегий 'precision'/'recall'.
        use_reconstruction : bool, optional
            Если True — вычислять MSE реконструкции (для автоэнкодеров).
            Если False — использовать predict_scores() напрямую.
            Если None — автоопределение по наличию метода 'predict' vs 'predict_scores'.
            
        Returns
        -------
        dict : {'threshold', 'metrics', 'results_df', 'scores'}
        """
        
        # Подготовка данных
        if isinstance(X_val, pd.DataFrame):
            if feature_names:
                X_val_array = X_val[feature_names].values
            else:
                X_val_array = X_val.select_dtypes(include=[np.number]).values
        else:
            X_val_array = X_val
        
        # Нормализация меток
        normal_label = split_info.get('normal_label', 'Norm')
        if hasattr(y_val, 'dtype') and (y_val.dtype == object or y_val.dtype == str):
            y_val_binary = (y_val == normal_label).astype(int).values
        else:
            y_val_binary = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
        
        # 🔍 Автоопределение типа модели
        if use_reconstruction is None:
            # Если у модели есть метод 'predict' и нет 'predict_scores' → вероятно, автоэнкодер
            # Если есть 'predict_scores' → используем его напрямую
            use_reconstruction = (
                hasattr(model, 'predict') and 
                not hasattr(model, 'predict_scores') and
                hasattr(model, 'model')  # Keras-модель внутри
            )
        
        # 🔧 Вычисление скоров аномальности
        if use_reconstruction:
            # === Для автоэнкодеров: MSE реконструкции ===
            try:
                X_val_recon = model.predict(X_val_array, verbose=0)
                # Проверяем, что реконструкция имеет ту же форму, что и вход
                if X_val_recon.ndim == 2 and X_val_array.ndim == 2:
                    scores = np.mean(np.square(X_val_array - X_val_recon), axis=1)
                else:
                    # Если форма не совпадает, пробуем predict_scores как fallback
                    logging.warning("⚠️ Форма реконструкции не совпадает с входом. Переключаюсь на predict_scores().")
                    scores = model.predict_scores(X_val_array)
            except Exception as e:
                logging.warning(f"⚠️ Ошибка при вычислении реконструкции: {e}. Переключаюсь на predict_scores().")
                scores = model.predict_scores(X_val_array)
        else:
            # === Для статистических моделей: используем predict_scores напрямую ===
            if hasattr(model, 'predict_scores'):
                scores = model.predict_scores(X_val_array)
            else:
                # Fallback: пробуем predict и считаем "скор" как вероятность аномалии
                pred = model.predict(X_val_array)
                # Если predict возвращает -1/1 (как sklearn), конвертируем
                if set(np.unique(pred)).issubset({-1, 1}):
                    scores = (pred == -1).astype(float)  # -1 = аномалия → высокий скор
                else:
                    scores = pred.astype(float)
        
        # 🔧 Защита от NaN/Inf в скорах
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            logging.warning("⚠️ Обнаружены NaN/Inf в скорах. Заменяю на медиану.")
            scores = np.nan_to_num(scores, nan=np.nanmedian(scores), posinf=np.nanmax(scores[scores < np.inf]), neginf=np.nanmin(scores[scores > -np.inf]))
        
        # Сбор результатов
        results_df = pd.DataFrame({
            'score': scores,
            'true_class': y_val_binary,
            'true_label': y_val.values if hasattr(y_val, 'values') else y_val
        })
        
        # 🔍 Перебор порогов
        # Используем перцентили для скорости и стабильности
        candidate_thresholds = np.percentile(scores, np.linspace(5, 95, 200))
        
        best_threshold, best_score = None, -1
        
        for thr in candidate_thresholds:
            # Предсказание: скор >= порог → аномалия (1), иначе норма (0)
            # Примечание: для Z-score высокий скор = аномалия, для некоторых моделей может быть наоборот
            pred_class = (scores >= thr).astype(int)
            
            # Пропускаем вырожденные случаи
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
                score = 1 - abs(precision - recall) if min(precision, recall) > 0 else -1
            else:
                score = f1
            
            if score > best_score:
                best_score, best_threshold = score, thr
        
        if best_threshold is None:
            # Fallback: медианный порог
            best_threshold = np.median(scores)
            logging.warning(f"⚠️ Не удалось подобрать порог. Использую медиану: {best_threshold}")
        
        # 🔧 Финальные метрики
        final_pred = (scores >= best_threshold).astype(int)
        final_metrics = {
            'precision': precision_score(y_val_binary, final_pred, zero_division=0),
            'recall': recall_score(y_val_binary, final_pred, zero_division=0),
            'f1': f1_score(y_val_binary, final_pred, zero_division=0),
            'accuracy': (final_pred == y_val_binary).mean(),
            'roc_auc': roc_auc_score(y_val_binary, scores),
            'threshold': best_threshold,
            'n_predictions': {
                'predicted_normal': int((final_pred == 0).sum()),
                'predicted_anomaly': int((final_pred == 1).sum()),
                'true_normal': int(y_val_binary.sum()),
                'true_anomaly': int((1 - y_val_binary).sum())
            }
        }
        
        results_df['pred_class'] = final_pred
        results_df['is_correct'] = (final_pred == y_val_binary).astype(int)
        
        logging.info(f"✓ Порог: {best_threshold:.6f}, F1={final_metrics['f1']:.4f}, ROC-AUC={final_metrics['roc_auc']:.4f}")
        
        return {
            'threshold': float(best_threshold),
            'metrics': final_metrics,
            'results_df': results_df,
            'scores': scores  # Возвращаем скоры для отладки/визуализации
        }