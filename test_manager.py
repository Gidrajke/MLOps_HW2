from sklearn.datasets import load_iris
import pandas as pd
from app.model_manager import ModelManager

# Загружаем данные
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Создаём менеджер
manager = ModelManager()

# 1. Проверяем список доступных моделей
print("Доступные модели:", manager.get_available_models())

# 2. Обучаем логистическую регрессию
print("\n--- Обучение логистической регрессии ---")
result = manager.train_model("logistic_regression", "iris_logreg", X, y, max_iter=200)
print(result)

# 3. Предсказание
print("\n--- Предсказание ---")
preds = manager.predict("iris_logreg", X.iloc[:5])
print(preds)

# 4. Удаление модели
print("\n--- Удаление модели ---")
res = manager.delete_model("iris_logreg")
print(res)
