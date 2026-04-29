import numpy as np

from typing import Optional
from sklearn.ensemble import IsolationForest
from .basedetector import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest: аномалии изолируются быстрее.
    """
    
    def __init__(
            self, 
            contamination: float = 0.1, 
            n_estimators: int = 100, 
            random_state: int = 42, 
            **kwargs):
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

# ======================================================
    def fit(self, 
            X_train: np.ndarray, 
            y_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> 'IsolationForestDetector':
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            **self.kwargs
        )

        self.model.fit(X_train)
        
        return self
    
# ======================================================
    def predict(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)