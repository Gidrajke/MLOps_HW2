from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as PydanticModel
from typing import List, Optional, Dict, Any
import pandas as pd
import io
from app.logger import logger
from app.model_manager import ModelManager
from app.s3_client import minio_client, BUCKET_NAME  # 👈 добавили работу с MinIO

app = FastAPI(
    title="ML Model Management API",
    description="API для обучения, хранения и предсказания ML-моделей + интеграция с MinIO",
    version="0.2.0",
)


manager = ModelManager()

class TrainRequest(PydanticModel):
    model_type: str
    model_name: str
    data: List[Dict[str, Any]]  # список объектов с фичами
    target: List[Any]           # список таргетов
    params: Optional[Dict[str, Any]] = {}  # гиперпараметры модели

class PredictRequest(PydanticModel):
    model_name: str
    data: List[Dict[str, Any]]

class RetrainRequest(PydanticModel):
    model_name: str
    data: List[Dict[str, Any]]
    target: List[Any]
    params: Optional[Dict[str, Any]] = {}

class DeleteRequest(PydanticModel):
    model_name: str

@app.get("/status")
def status():
    logger.info("GET /status called")
    return {"status": "ok", "version": "0.2.0"}

@app.get("/models")
def list_models():
    """Возвращает список доступных типов моделей"""
    logger.info("GET /models called")
    return {"available_models": manager.get_available_models()}

@app.post("/train")
def train_model(request: TrainRequest):
    logger.info(f"POST /train called for {request.model_name} ({request.model_type})")
    try:
        X = pd.DataFrame(request.data)
        y = pd.Series(request.target)
        result = manager.train_model(request.model_type, request.model_name, X, y, **request.params)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_model(request: PredictRequest):
    logger.info(f"POST /predict called for {request.model_name}")
    try:
        X = pd.DataFrame(request.data)
        preds = manager.predict(request.model_name, X)
        return {"status": "success", "predictions": preds}
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    logger.info(f"POST /retrain called for {request.model_name}")
    try:
        X = pd.DataFrame(request.data)
        y = pd.Series(request.target)
        result = manager.retrain_model(request.model_name, X, y, **request.params)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Ошибка при переобучении: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/delete")
def delete_model(request: DeleteRequest):
    logger.info(f"DELETE /delete called for {request.model_name}")
    try:
        result = manager.delete_model(request.model_name)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Ошибка при удалении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Загрузка файла (датасета или модели) в MinIO
    """
    try:
        file_data = await file.read()
        minio_client.put_object(
            BUCKET_NAME,
            file.filename,
            io.BytesIO(file_data),
            length=len(file_data),
            content_type=file.content_type
        )
        logger.info(f"Файл {file.filename} успешно загружен в MinIO.")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Ошибка загрузки файла в MinIO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Скачивание файла из MinIO
    """
    try:
        response = minio_client.get_object(BUCKET_NAME, filename)
        logger.info(f"Файл {filename} скачан из MinIO.")
        return StreamingResponse(response, media_type="application/octet-stream")
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {e}")
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден")
