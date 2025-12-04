from abc import ABC, abstractmethod
from pathlib import Path
import joblib
from typing import Any

class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех ML-моделей.
    """

    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self.model = model

    @abstractmethod
    def train(self, X, y, **kwargs):
        """Обучение модели"""
        pass

    @abstractmethod
    def predict(self, X):
        """Предсказание"""
        pass

    def save(self, path: Path):
        """Сохранить модель на диск"""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return str(path)

    def load(self, path: Path):
        """Загрузить модель с диска"""
        self.model = joblib.load(path)
        return self.model

    def get_params(self) -> dict[str, Any]:
        """Получить параметры модели"""
        return self.model.get_params() if self.model else {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.model_name})"
