# src/models/zscore_detector.py
import numpy as np
import logging

from typing import Optional, Union
from ..autoencoders.basedetector import BaseAnomalyDetector
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
            # res = np.nanmax(z_scores, axis=1)
            # logging.info(f"Z scores max result: \n{res}")
            return z_scores
        elif self.aggregation == 'mean':
            return np.mean(z_scores, axis=1)
        elif self.aggregation == 'l2':
            return np.sqrt(np.sum(z_scores**2, axis=1))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
