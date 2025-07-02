import re
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
import uvicorn
import xgboost as xgb
import numpy as np
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import datetime
import os
import pickle
import asyncio
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for request/response based on API schemas
class Signal(BaseModel):
    filename: str
    values: List[float]

class Discharge(BaseModel):
    id: str
    signals: List[Signal]
    times: List[float]
    length: int
    anomalyTime: Optional[float] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    executionTimeMs: float
    model: str
    details: Optional[Dict[str, Any]] = None

class StartTrainingRequest(BaseModel):
    totalDischarges: int = Field(..., ge=1)
    timeoutSeconds: int = Field(..., ge=1)

class StartTrainingResponse(BaseModel):
    expectedDischarges: int = Field(..., ge=0)

class DischargeAck(BaseModel):
    ordinal: int = Field(..., ge=1)
    totalDischarges: int = Field(..., ge=1)

class TrainingMetrics(BaseModel):
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1Score: Optional[float] = None

class TrainingResponse(BaseModel):
    status: str
    message: Optional[str] = None
    trainingId: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    executionTimeMs: float

class HealthCheckResponse(BaseModel):
    name: str
    uptime: float
    lastTraining: Optional[str]

class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Initialize FastAPI application
app = FastAPI(
    title="XGBoost Anomaly Detection API",
    description="API for anomaly detection using XGBoost models",
    version="1.0.0"
)

# Global variables
MODEL_PATH = "xgboost_model.pkl"
start_time = time.time()
last_training_time = None
model = None
# Training session state
training_buffer: List[Discharge] = []
expected_discharges: Optional[int] = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        logger.info("Existing model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

# Helper functions
def extract_features(window: list[float]) -> np.ndarray:
    """Extract features from discharge data for model training/prediction"""
    features = []
        
    # Extract statistical features
    features.extend([
        np.mean(window),
        np.std(window),
        np.min(window),
        np.max(window),
        np.median(window)
    ])
    
    return np.array(features).reshape(1, -1)

def sliding_window(values: list[float], window_size: int = 16, overlap: float = 0.0) -> list[list[float]]:
    """Generate sliding windows over the values"""
    step = int(window_size * (1 - overlap))
    windows = []

    for i in range(0, len(values) - window_size + 1, step):
        windows.append(values[i:i + window_size])
    
    # Return remaining values
    if len(values) % step != 0:
        remaining = values[-(len(values) % step):]
        if len(remaining) > 0:
            windows.append(remaining)

    if len(values) < window_size:
        # values are shorter than window size
        return [values]

    return windows

def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

def train_from_discharges(discharges: List[Discharge]) -> float:
    """Train XGBoost model using a list of discharges.
    Returns the execution time in milliseconds."""
    global model

    start_execution = time.time()

    X, y = [], []
    for discharge in discharges:
        for signal in discharge.signals:
            if len(signal.values) == 0:
                continue
            windows = sliding_window(signal.values, window_size=48, overlap=0.5)
            for window in windows:
                features = extract_features(window)
                X.append(features[0])
                y.append(1 if is_anomaly(discharge) else 0)

    X_array = np.array(X)
    y_array = np.array(y)

    # We should apply scale_pos_weight as the dataset is likely imbalanced
    pos_count = np.sum(y_array == 1)
    neg_count = np.sum(y_array == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Training with {len(X_array)} samples, {len(y_array)} labels, scale_pos_weight={scale_pos_weight}")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['aucpr', 'logloss'],
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 8,
        'eta': 0.02,
        'gamma': 1.0,
        'min_child_weight': 3,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
    }

    dtrain = xgb.DMatrix(X_array, label=y_array)
    model = xgb.train(params, dtrain, num_boost_round=10000)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return (time.time() - start_execution) * 1000

async def train_async(discharges: List[Discharge]):
    """Run training in a background thread so the API stays responsive."""
    global expected_discharges, training_buffer, last_training_time
    try:
        await asyncio.to_thread(train_from_discharges, discharges)
        last_training_time = datetime.datetime.now().isoformat()
    finally:
        training_buffer = []
        expected_discharges = None

@app.post("/train", response_model=StartTrainingResponse)
async def start_training(request: StartTrainingRequest):
    global expected_discharges, training_buffer
    if expected_discharges is not None:
        raise HTTPException(status_code=503, detail="Node is busy or cannot train now")
    expected_discharges = request.totalDischarges
    training_buffer = []
    return StartTrainingResponse(expectedDischarges=expected_discharges)

@app.post("/train/{ordinal}", response_model=DischargeAck)
async def push_discharge(ordinal: int, discharge: Discharge, background_tasks: BackgroundTasks):
    global expected_discharges, training_buffer
    if expected_discharges is None:
        raise HTTPException(status_code=400, detail="Training session not started")
    if ordinal != len(training_buffer) + 1 or ordinal > expected_discharges:
        raise HTTPException(status_code=400, detail="Invalid ordinal")
    training_buffer.append(discharge)
    ack = DischargeAck(ordinal=ordinal, totalDischarges=expected_discharges)
    if ordinal == expected_discharges:
        # Start training asynchronously so the API responds immediately
        background_tasks.add_task(train_async, training_buffer.copy())
    return ack

@app.post("/predict", response_model=PredictionResponse)
async def predict(discharge: Discharge):
    global model
    
    start_execution = time.time()
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Model not trained",
                code="MODEL_NOT_FOUND",
                details={"message": "Please train the model first"}
            ).dict()
        )
    
    try:
        # Check if there are discharges to process
            
        # Process all discharges for prediction
        all_predictions = []
        all_confidences = []
        wheighted_predictions = []
        for signal in discharge.signals:

                if len(signal.values) == 0:
                    continue
                
                # Generate sliding windows for the signal values
                windows = sliding_window(signal.values, window_size=48, overlap=0.5)
                
                # Extract features from each window and predict for each
                i = 0
                for window in windows:
                    i += 1
                    features = extract_features(window)

                    # Convert to DMatrix
                    dtest = xgb.DMatrix(features)
            
                    # Make prediction
                    prediction_probability = model.predict(dtest)[0]
                    prediction = 1 if prediction_probability > 0.5 else 0
                    weigthted_prediction = prediction_probability if prediction == 1 else -1 + prediction_probability
                    wheighted_predictions.append(weigthted_prediction)

                    all_predictions.append(prediction)
                    all_confidences.append(float(prediction_probability if prediction == 1 else 1 - prediction_probability))

        # Determine overall prediction
        # Use majority vote for final prediction
        final_prediction = 1 if sum(wheighted_predictions) > 0 else 0
        
        # Use average confidence
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        # Calculate feature importance if available
        feature_importance = None
        try:
            importance = model.get_score(importance_type='weight')
            feature_importance = list(importance.values())
        except:
            pass
            
        # Calculate execution time
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        return PredictionResponse(
            prediction="Anomaly" if final_prediction == 1 else "Normal",
            confidence=float(avg_confidence),
            executionTimeMs=execution_time,
            model="xgboost",
            details={
                "individualPredictions": all_predictions,
                "individualConfidences": all_confidences,
                "numDischargesProcessed": 1,
                "featureImportance": feature_importance
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Prediction failed",
                code="PREDICTION_ERROR",
                details={"message": str(e)}
            ).dict()
        )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        name="xgboost",
        uptime=time.time() - start_time,
        lastTraining=last_training_time,
    )

# Custom middleware to handle large request JSON payloads
@app.middleware("http")
async def increase_json_size_limit(request: Request, call_next):
    # Increase JSON size limit for this specific request
    # Default is 1MB, we're setting it to 64MB
    request.app.state.json_size_limit = 64 * 1024 * 1024  # 64MB
    response = await call_next(request)
    return response

if __name__ == "__main__":
    # Set server settings for large JSON payloads
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True, 
                limit_concurrency=50, 
                limit_max_requests=20000,
                timeout_keep_alive=120)

