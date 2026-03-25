


from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseAnomalyDetector(ABC):
    """
    Единый интерфейс для всех моделей детекции аномалий.
    Все модели должны реализовывать эти три метода.
    """
    
    @abstractmethod
    def get_model(self):
        pass


    @abstractmethod
    def fit(
        self, 
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None):
        
        pass


    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает 'score аномальности' для каждого образца.
        Чем выше значение — тем более аномальным считается образец.
        Для автоэнкодеров: MSE реконструкции.
        Для sklearn-моделей: отрицательный decision_function.
        """
        pass


    @abstractmethod
    def get_name(self) -> str:
        """Название модели для отчётов и логирования."""
        pass