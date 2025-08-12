from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
import uvicorn
import xgboost as xgb
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import datetime
import os
import pickle
import asyncio
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

class WindowProperties(BaseModel):
    featureValues: List[float] = Field(..., min_items=1)
    prediction: str = Field(..., pattern=r'^(Anomaly|Normal)$')
    justification: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    executionTimeMs: float
    model: str
    windowSize: int = 48
    windows: List[WindowProperties]

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
WINDOW_SIZE = 16
SAMPLING_TIME = 2e-3  # 2ms
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

    if len(window) == 0:
        raise ValueError("Window cannot be empty")

    features = []

    # Features are grouped in:
    # - Core statistics: mean, slope, RMS
    # - Dynamic features: maximun slope, 2nd derivative
    # - Shape: skewness, kurtosis. Not implemented yet.

    # helper variables
    mean = np.mean(window)
    if len(window) > 1:
        diff = np.diff(window)
    else:
        diff = np.array([0.0])

    if len(window) > 2:
        abs_second_derivate = np.abs(np.diff(diff))
    else:
        abs_second_derivate = np.array([0.0])

    features.extend([
        # Core statistics
        mean,                                                             # Mean
        diff.mean() * 1/SAMPLING_TIME,                                    # Slope (mean of differences) 
        np.log1p(np.sqrt(np.mean(np.square(window)))),                    # Log-RMS

        # Dynamic features
        np.max(np.abs(diff)),                                             # Max slope
        np.min(abs_second_derivate) * 1/(SAMPLING_TIME**2),               # Min 2nd derivative
        np.max(abs_second_derivate) * 1/(SAMPLING_TIME**2),               # Max 2nd derivative
        np.mean(abs_second_derivate) * 1/(SAMPLING_TIME**2),              # Mean 2nd derivative

        # Shape features: skewness and kurtosis
        # TODO: use scipy.stats for skewness and kurtosis if more complex features are needed
    ])
    return np.array(features).reshape(1, -1)

def sliding_window(values: list[float], window_size: int = WINDOW_SIZE, overlap: float = 0.0) -> list[list[float]]:
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
    num_tendency = 3  # We will add the past 3 features vectors as a tendency
    feature_size = 7  # Number of features per window
    
    if len(discharges) < num_tendency:
        raise ValueError(f"Not enough discharges to train the model, expected at least {num_tendency}, got {len(discharges)}")
    
    for discharge in discharges:
        # Group windows from all signals by their time index
        aligned_windows = {}
        signal_indices = {}
        
        # First, collect windows from all signals
        for signal_idx, signal in enumerate(discharge.signals):
            if len(signal.values) == 0:
                continue
                
            windows = sliding_window(signal.values)
            for window_idx, window in enumerate(windows):
                if window_idx not in aligned_windows:
                    aligned_windows[window_idx] = []
                    signal_indices[window_idx] = []
                aligned_windows[window_idx].append(window)
                signal_indices[window_idx].append(signal_idx)
        
        # Process aligned windows
        for window_idx in sorted(aligned_windows.keys()):
            if window_idx < num_tendency:
                continue
                
            # Average features across all signals at this time point
            current_window_features = np.zeros(feature_size)
            for window in aligned_windows[window_idx]:
                current_window_features += extract_features(window)[0]
            
            if len(aligned_windows[window_idx]) > 0:
                current_window_features /= len(aligned_windows[window_idx])
            
            # Create feature vector with consistent size
            all_features = [current_window_features]
                
            # Add tendency features (also averaged across signals)
            for j in range(1, num_tendency + 1):
                prev_idx = window_idx - j
                if prev_idx in aligned_windows and len(aligned_windows[prev_idx]) > 0:
                    prev_window_features = np.zeros(feature_size)
                    for window in aligned_windows[prev_idx]:
                        prev_window_features += extract_features(window)[0]
                    prev_window_features /= len(aligned_windows[prev_idx])
                    all_features.append(prev_window_features)
                else:
                    # If no previous window, use zeros
                    all_features.append(np.zeros(feature_size))
            
            # Flatten and append to feature matrix
            X.append(np.concatenate(all_features))
            y.append(1 if is_anomaly(discharge) else 0)
            
        logger.info(f"Extracted features for discharge {discharge.id}")

    X_array = np.array(X)
    y_array = np.array(y)

    # We should apply scale_pos_weight as the dataset is likely imbalanced
    pos_count = np.sum(y_array == 1)
    neg_count = np.sum(y_array == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Training with {len(X_array)} samples, {len(y_array)} labels, scale_pos_weight={scale_pos_weight}")

    params = {
        # Algorithm and hardware settings
        'tree_method': 'hist',
        'n_jobs': -1,
        # Objective and evaluation metrics
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr', # 'logloss'
        # Regularization and overfitting control
        'learning_rate': 0.05,
        'scale_pos_weight': scale_pos_weight,
        # Hyperparameters
        'max_depth': 6,
        'min_child_weight': 3,
        'gamma': 1.0,
        'eta': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # incremental learning
        # 'process_type': 'update',
        # 'refresh_leaf': true,
    }

    dtrain = xgb.DMatrix(X_array, label=y_array)
    model = xgb.train(params, 
                      dtrain, 
                      num_boost_round=10000, 
                    #   early_stopping_rounds=50 # To enable early stopping, we need a validation set
                      verbose_eval=10
                      )

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
            ).model_dump()
        )
    
    try:
        # Group windows from all signals by their time index
        aligned_windows = {}
        all_windows = []  # To store all windows for response
        all_predictions = []
        all_confidences = []
        wheighted_predictions = []
        feature_size = 7
        num_tendency = 3
        
        # First, collect windows from all signals
        for signal_idx, signal in enumerate(discharge.signals):
            if len(signal.values) == 0:
                continue
                
            windows = sliding_window(signal.values)
            all_windows.extend(windows)  # Store windows for response
            
            for window_idx, window in enumerate(windows):
                if window_idx not in aligned_windows:
                    aligned_windows[window_idx] = []
                aligned_windows[window_idx].append(window)
        
        # Process aligned windows for prediction
        for window_idx in sorted(aligned_windows.keys()):
            if window_idx < num_tendency:
                continue
                
            # Average features across all signals at this time point
            current_window_features = np.zeros(feature_size)
            for window in aligned_windows[window_idx]:
                current_window_features += extract_features(window)[0]
            
            if len(aligned_windows[window_idx]) > 0:
                current_window_features /= len(aligned_windows[window_idx])
            
            # Create feature vector with consistent size (matching training)
            all_features = [current_window_features]
                
            # Add tendency features (also averaged across signals)
            for j in range(1, num_tendency + 1):
                prev_idx = window_idx - j
                if prev_idx in aligned_windows and len(aligned_windows[prev_idx]) > 0:
                    prev_window_features = np.zeros(feature_size)
                    for window in aligned_windows[prev_idx]:
                        prev_window_features += extract_features(window)[0]
                    prev_window_features /= len(aligned_windows[prev_idx])
                    all_features.append(prev_window_features)
                else:
                    # If no previous window, use zeros
                    all_features.append(np.zeros(feature_size))
            
            # Flatten and prepare for prediction
            features_array = np.concatenate(all_features).reshape(1, -1)
            dtest = xgb.DMatrix(features_array)
            
            # Make prediction
            prediction_probability = model.predict(dtest)[0]
            prediction = 1 if prediction_probability > 0.5 else 0
            weigthted_prediction = prediction_probability if prediction == 1 else -1 + prediction_probability
            wheighted_predictions.append(weigthted_prediction)

            all_predictions.append(prediction)
            all_confidences.append(float(prediction_probability if prediction == 1 else 1 - prediction_probability))
            
        # Determine overall prediction
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

        # DEBUG: Use matplotlib to visualize the feature importance and confidences
        # import matplotlib.pyplot as plt

        # if feature_importance:
        #     plt.figure(figsize=(10, 6))
        #     plt.barh(range(len(feature_importance)), feature_importance, align='center')
        #     plt.xlabel('Feature Importance')
        #     plt.ylabel('Features')
        #     plt.title('XGBoost Feature Importance')
        #     plt.show()

        # if all_confidences:
        #     plt.figure(figsize=(10, 6))
        #     plt.hist(all_confidences, bins=20, alpha=0.7)
        #     plt.xlabel('Confidence')
        #     plt.ylabel('Frequency')
        #     plt.title('Prediction Confidence Distribution')
        #     plt.show()

        return PredictionResponse(
            prediction="Anomaly" if final_prediction == 1 else "Normal",
            confidence=float(avg_confidence) if final_prediction == 1 else 1 - float(avg_confidence),
            executionTimeMs=execution_time,
            model="xgboost",
            windowSize=48,
            windows=[
                WindowProperties(
                    featureValues=[float(x) for x in extract_features(window)[0].tolist()],
                    prediction="Anomaly" if pred == 1 else "Normal",
                    justification=float(conf) if pred == 1 else 1 - float(conf)
                ) for window, pred, conf in zip(windows, all_predictions, all_confidences)
            ]
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
    uvicorn.run("main:app", host="0.0.0.0", port=8003, timeout_keep_alive=120)

