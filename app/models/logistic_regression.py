from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from app.models.base import BaseModel

class LogisticRegressionModel(BaseModel):
    def train(self, X, y, **kwargs):
        test_size = kwargs.get("test_size", 0.2)
        random_state = kwargs.get("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # передаём остальные гиперпараметры модели
        model_params = {k: v for k, v in kwargs.items() if k not in ["test_size", "random_state"]}
        self.model = LogisticRegression(**model_params)
        self.model.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))
        return {"status": "trained", "accuracy": acc}

    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(X).tolist()
