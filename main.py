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
from sklearn.preprocessing import StandardScaler
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
SCALER_PATH = "feature_scaler.pkl"
start_time = time.time()
last_training_time = None
model = None
scaler = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        logger.info("Existing model loaded successfully")
        if os.path.exists(SCALER_PATH):
            scaler = pickle.load(open(SCALER_PATH, "rb"))
            logger.info("Existing scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

# Helper functions
def extract_features(discharge: Discharge) -> np.ndarray:
    """Extract features from discharge data for model training/prediction"""
    features = []
    
    # Get basic statistics for each signal
    for signal in discharge.signals:
        values = np.array(signal.values)
        if len(values) == 0:
            continue
        
        # Extract statistical features
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values)
        ])
        
        # # Add some time-based features if times are available
        # if discharge.times and len(discharge.times) == len(values):
        #     times = np.array(discharge.times)
        #     # Calculate rate of change features
        #     if len(times) > 1:
        #         dv_dt = np.diff(values) / np.diff(times)
        #         features.extend([
        #             np.mean(dv_dt),
        #             np.std(dv_dt),
        #             np.max(np.abs(dv_dt))
        #         ])
    
    return np.array(features).reshape(1, -1)

def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    global model, scaler, last_training_time
    
    start_execution = time.time()
    
    try:
        # Extract features and labels from training data
        X = []
        y = []
        
        for discharge in request.discharges:
            features = extract_features(discharge)
            label = 1 if is_anomaly(discharge) else 0
            
            X.append(features.flatten())  # Flatten for compatibility
            y.append(label)
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Training data shape: {X_array.shape}, Labels: {np.sum(y_array)} anomalies out of {len(y_array)}")
        
        # Initialize and fit the scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        
        # Save the scaler
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
            
        # Set up XGBoost parameters (can be overridden by options)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'seed': 42
        }
        
        # Apply custom hyperparameters if provided
        if request.options and request.options.hyperparameters:
            for param, value in request.options.hyperparameters.items():
                params[param] = value
        
        # Convert data to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_scaled, label=y_array)
        
        # Train the model
        num_rounds = 100  # Default rounds
        if request.options and request.options.epochs:
            num_rounds = request.options.epochs
            
        model = xgb.train(params, dtrain, num_rounds)
        
        # Save the trained model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
            
        # Update last training time
        last_training_time = datetime.datetime.now().isoformat()
        
        # Calculate metrics
        y_pred_proba = model.predict(dtrain)
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_proba]
        
        correct = sum(1 for i in range(len(y_array)) if y_pred[i] == y_array[i])
        accuracy = correct / len(y_array) if len(y_array) > 0 else 0
        
        # Calculate execution time
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        # Generate training ID
        training_id = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TrainingResponse(
            status="success",
            message="Training completed successfully",
            trainingId=training_id,
            metrics=TrainingMetrics(
                accuracy=accuracy,
                loss=model.attributes().get('best_score', 0.0),
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
    global model, scaler
    
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
            
        # Process all discharges for prediction (not just the first one)
        all_predictions = []
        all_confidences = []
        
        for discharge in request.discharges:
            features = extract_features(discharge)
            
            # Scale features
            if scaler:
                features = scaler.transform(features)
            
            # Convert to DMatrix
            dtest = xgb.DMatrix(features)
            
            # Make prediction
            prediction_probability = model.predict(dtest)[0]
            prediction = 1 if prediction_probability > 0.5 else 0
            
            all_predictions.append(prediction)
            all_confidences.append(float(prediction_probability if prediction == 1 else 1 - prediction_probability))
        
        # Determine overall prediction
        # Use majority vote for final prediction
        final_prediction = 1 if sum(all_predictions) > len(all_predictions) / 2 else 0
        
        # Use average confidence
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
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
                limit_concurrency=20, 
                limit_max_requests=20000,
                timeout_keep_alive=120)

