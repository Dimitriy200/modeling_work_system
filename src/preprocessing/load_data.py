# ======================================================
# Интерфейс для loader-ов данных
# ======================================================

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class LoadData(ABC):
    """
    Интерфейс для всех загрузчиков данных.
    Гарантирует наличие метода load(), возвращающего pd.DataFrame или np.ndarray.
    """
    @abstractmethod
    def data_raw_load(
            self, 
            directory_input_path: str,
            directory_out_path: str = None ) -> pd.DataFrame | None:
        
        pass