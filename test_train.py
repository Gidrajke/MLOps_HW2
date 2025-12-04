import requests

url = "http://127.0.0.1:8000/train"

payload = {
    "model_type": "logistic_regression",
    "model_name": "iris_logreg",
    "data": [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 4.9, "sepal width (cm)": 3.0, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.0, "sepal width (cm)": 2.9, "petal length (cm)": 4.5, "petal width (cm)": 1.5},
        {"sepal length (cm)": 5.5, "sepal width (cm)": 2.5, "petal length (cm)": 4.0, "petal width (cm)": 1.3},
        {"sepal length (cm)": 5.0, "sepal width (cm)": 3.6, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    ],
    "target": [0, 0, 1, 1, 0],
    "params": {"max_iter": 200}
}

resp = requests.post(url, json=payload)
print(resp.json())
