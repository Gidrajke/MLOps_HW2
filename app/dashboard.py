import io
import streamlit as st
import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"

st.title("ML Model Dashboard")

action = st.selectbox("Выберите действие", ["Train", "Predict", "Retrain", "Delete"])

if action == "Train":
    model_name = st.text_input("Имя модели", "iris_logreg")
    model_type = st.selectbox("Тип модели", ["logistic_regression", "decision_tree"])
    max_iter = st.number_input("max_iter", 100, 1000, 200)

    st.write("Введите данные (CSV)")
    data_csv = st.text_area("data",
                            "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5.1,3.5,1.4,0.2\n6.0,2.9,4.5,1.5")
    target_csv = st.text_area("target", "0,1")

    if st.button("Train"):
        data = pd.read_csv(io.StringIO(data_csv))
        target = list(map(int, target_csv.strip().split(",")))
        payload = {
            "model_type": model_type,
            "model_name": model_name,
            "data": data.to_dict(orient="records"),
            "target": target,
            "params": {"max_iter": max_iter}
        }
        resp = requests.post(f"{BASE_URL}/train", json=payload)
        st.json(resp.json())

elif action == "Predict":
    model_name = st.text_input("Имя модели", "iris_logreg")
    st.write("Введите данные для предсказания (CSV)")
    data_csv = st.text_area("data",
                            "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5.1,3.5,1.4,0.2")
    if st.button("Predict"):
        data = pd.read_csv(io.StringIO(data_csv))
        payload = {"model_name": model_name, "data": data.to_dict(orient="records")}
        resp = requests.post(f"{BASE_URL}/predict", json=payload)
        st.json(resp.json())

elif action == "Delete":
    model_name = st.text_input("Имя модели", "iris_logreg")
    if st.button("Delete"):
        payload = {"model_name": model_name}
        resp = requests.delete(f"{BASE_URL}/delete", json=payload)
        st.json(resp.json())
