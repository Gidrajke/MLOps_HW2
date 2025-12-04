from typing import Dict, Any, Type
from pathlib import Path
from app.models.base import BaseModel
from app.models.logistic_regression import LogisticRegressionModel
from app.models.random_forest import RandomForestModel
from app.logger import logger

# Папка, куда будем сохранять модели
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

class ModelManager:
    """
    Менеджер для управления ML-моделями:
    - хранение, обучение, предсказания, удаление
    """

    def __init__(self):
        # хранение активных моделей в памяти
        self.trained_models: Dict[str, BaseModel] = {}
        # список доступных типов моделей
        self.available_model_classes: Dict[str, Type[BaseModel]] = {
            "logistic_regression": LogisticRegressionModel,
            "random_forest": RandomForestModel,
        }

    def get_available_models(self):
        """Возвращает список доступных типов моделей"""
        return list(self.available_model_classes.keys())

    def train_model(self, model_type: str, model_name: str, X, y, **kwargs):
        """Создаёт и обучает новую модель"""
        if model_type not in self.available_model_classes:
            raise ValueError(f"Модель '{model_type}' не найдена")

        model_class = self.available_model_classes[model_type]
        model_instance = model_class(model_name)
        result = model_instance.train(X, y, **kwargs)

        self.trained_models[model_name] = model_instance
        model_path = MODELS_DIR / f"{model_name}.joblib"
        model_instance.save(model_path)

        logger.info(f"Модель {model_name} ({model_type}) обучена и сохранена в {model_path}")
        return result

    def predict(self, model_name: str, X):
        """Делает предсказание обученной моделью"""
        if model_name not in self.trained_models:
            model_path = MODELS_DIR / f"{model_name}.joblib"
            if not model_path.exists():
                raise ValueError(f"Модель '{model_name}' не найдена")
            # загружаем с диска
            for model_type, cls in self.available_model_classes.items():
                try:
                    model_instance = cls(model_name)
                    model_instance.load(model_path)
                    self.trained_models[model_name] = model_instance
                    break
                except Exception:
                    continue
        model = self.trained_models[model_name]
        return model.predict(X)

    def retrain_model(self, model_name: str, X, y, **kwargs):
        """Переобучает уже существующую модель"""
        if model_name not in self.trained_models:
            raise ValueError(f"Модель '{model_name}' не найдена в памяти")

        model = self.trained_models[model_name]
        result = model.train(X, y, **kwargs)
        model.save(MODELS_DIR / f"{model_name}.joblib")
        logger.info(f"Модель {model_name} переобучена.")
        return result

    def delete_model(self, model_name: str):
        """Удаляет модель из памяти и с диска"""
        if model_name in self.trained_models:
            del self.trained_models[model_name]

        model_path = MODELS_DIR / f"{model_name}.joblib"
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Модель {model_name} удалена.")
            return {"status": "deleted"}
        else:
            raise ValueError(f"Файл модели '{model_name}' не найден")
