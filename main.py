import re
from fastapi import FastAPI, HTTPException, Request
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
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for request/response based on API schemas
class Signal(BaseModel):
    fileName: str
    values: List[float]
    times: Optional[List[float]] = None
    length: Optional[int] = None

class Discharge(BaseModel):
    id: str
    times: Optional[List[float]] = None
    length: Optional[int] = None
    anomalyTime: Optional[float] = None
    signals: List[Signal]

class PredictionRequest(BaseModel):
    discharges: List[Discharge]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    executionTimeMs: float
    model: str
    details: Optional[Dict[str, Any]] = None

class TrainingOptions(BaseModel):
    epochs: Optional[int] = None
    batchSize: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    discharges: List[Discharge]
    options: Optional[TrainingOptions] = None

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

class MemoryInfo(BaseModel):
    total: float
    used: float

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    uptime: float
    memory: MemoryInfo
    load: float
    lastTraining: Optional[str] = None

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

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    global model, last_training_time
    
    start_execution = time.time()
    
    try:
        # Extract features and labels from training data
        X = []
        y = []
        
        for discharge in request.discharges:
            # Extract features from each signal in the discharge
            for signal in discharge.signals:
                if len(signal.values) == 0:
                    continue
                # Generate sliding windows for the signal values
                windows = sliding_window(signal.values, window_size=48, overlap=0.5)
                # Extract features from each window
                for window in windows:
                    features = extract_features(window)
                    X.append(features[0])
                    y.append(1 if is_anomaly(discharge) else 0)

        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Training data shape: {X_array.shape}, Labels: {np.sum(y_array)} anomalies out of {len(y_array)}")
            
        # Set up XGBoost parameters (can be overridden by options)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 8,                 # Aumenta el riesgo de overfitting, pero aporta expresividad
            'eta': 0.02,                    # Reduce overfitting. Tenemos muchos datos, por lo que nos podemos permitir un valor bajo
            'gamma': 1.0,                   # Valor medio, no necesitamos muchas divisiones, pero el modelo es complejo
            'min_child_weight': 3,          # En nuestro caso, los patrones no son nada sutiles. No nos hace falta fijarnos en los detalles
            'subsample': 1.0,               # Queremos aprovechar todo el dataset
            'colsample_bytree': 0.8,        # Reduce la correlacion entre arboles
        }
        
        # Apply custom hyperparameters if provided
        if request.options and request.options.hyperparameters:
            for param, value in request.options.hyperparameters.items():
                params[param] = value
        
        # Convert data to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_array, label=y_array)
        
        # Train the model
        num_rounds = 10000  # Default rounds
        if request.options and request.options.epochs:
            num_rounds = request.options.epochs
            
        model = xgb.train(params, dtrain, num_rounds)
        
        # Save the trained model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
            
        # Update last training time
        last_training_time = datetime.datetime.now().isoformat()
        
        # Calculate execution time
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        # Generate training ID
        training_id = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TrainingResponse(
            status="success",
            message="Training completed successfully",
            trainingId=training_id,
            metrics=TrainingMetrics(
                accuracy=0.85,  # Placeholder for actual accuracy
                f1Score=0.8,    # Placeholder for actual F1 score
                loss=0.1,       # Placeholder for actual loss
            ),
            executionTimeMs=execution_time
        )
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(
                error="Training failed",
                code="TRAINING_ERROR",
                details={"message": str(e)}
            ).dict()
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model
    
    start_execution = time.time()
    print(f"Received prediction request with {len(request.discharges)} discharges")
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
        if len(request.discharges) == 0:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="No discharge data provided",
                    code="INVALID_INPUT"
                ).dict()
            )
            
        # Process all discharges for prediction
        all_predictions = []
        all_confidences = []
        wheighted_predictions = []

        for discharge in request.discharges:
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
            prediction=final_prediction,
            confidence=float(avg_confidence),
            executionTimeMs=execution_time,
            model="xgboost",
            details={
                "individualPredictions": all_predictions,
                "individualConfidences": all_confidences,
                "numDischargesProcessed": len(request.discharges),
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
    # Get memory information
    mem = psutil.virtual_memory()
    
    return HealthCheckResponse(
        status="online" if model is not None else "degraded",
        version="1.0.0",
        uptime=time.time() - start_time,
        memory=MemoryInfo(
            total=mem.total / (1024*1024),  # Convert to MB
            used=mem.used / (1024*1024)
        ),
        load=psutil.cpu_percent() / 100,
        lastTraining=last_training_time
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

