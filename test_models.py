import pandas as pd
from sklearn.datasets import load_iris
from app.models.logistic_regression import LogisticRegressionModel
from app.models.random_forest import RandomForestModel

# Загружаем тестовые данные
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Logistic Regression
log_model = LogisticRegressionModel("log_reg")
print("Обучаем логистическую регрессию...")
res1 = log_model.train(X, y)
print(res1)
print("Предсказание:", log_model.predict(X.iloc[:5]))

# Random Forest
rf_model = RandomForestModel("rf")
print("\nОбучаем случайный лес...")
res2 = rf_model.train(X, y, n_estimators=50)
print(res2)
print("Предсказание:", rf_model.predict(X.iloc[:5]))
