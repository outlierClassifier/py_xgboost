# XGBoost Anomaly Detection API

A machine learning service for anomaly detection using XGBoost, designed for analyzing discharge data. This API aligns with the JSON schema definitions provided in the project.

## Features

- **Anomaly Detection**: Predict anomalies in discharge data using XGBoost models
- **Model Training**: Train models with custom data and hyperparameters
- **Health Monitoring**: Check the status and health of the service
- **API Schema Compliance**: Fully compatible with the defined API schemas

## Requirements

```
fastapi>=0.95.0
uvicorn>=0.21.1
xgboost>=1.7.5
numpy>=1.24.3
scikit-learn>=1.2.2
psutil>=5.9.5
python-multipart>=0.0.6
```

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install fastapi uvicorn xgboost numpy scikit-learn psutil python-multipart
```

## Getting Started

Run the server:

```bash
python main.py
```

The API will be accessible at `http://localhost:8003`. FastAPI also provides automatic documentation at:
- Swagger UI: `http://localhost:8003/docs`
- ReDoc: `http://localhost:8003/redoc`

## API Endpoints

### Train a Model

**POST /train**

Train the XGBoost model with discharge data.

Example request:
```json
{
  "discharges": [
    {
      "id": "74467",
      "times": [41.0520, 41.0540, 41.0560],
      "anomalyTime": 41.0562,
      "signals": [
        {
          "fileName": "DES_74467_01_r2_sliding.txt",
          "values": [-759337, -760461, -761585]
        },
        {
          "fileName": "DES_74467_02_r2_sliding.txt",
          "values": [273.4, 273.2, 272.9]
        }
      ]
    },
    {
      "id": "74468",
      "times": [41.0520, 41.0540, 41.0560],
      "signals": [
        {
          "fileName": "DES_74468_01_r2_sliding.txt",
          "values": [-759337, -760461, -761585]
        }
      ]
    }
  ],
  "options": {
    "epochs": 100,
    "hyperparameters": {
      "max_depth": 6,
      "eta": 0.15
    }
  }
}
```

### Predict Anomalies

**POST /predict**

Predict whether a discharge contains an anomaly.

Example request:
```json
{
  "discharges": [
    {
      "id": "74467",
      "times": [41.0520, 41.0540, 41.0560],
      "signals": [
        {
          "fileName": "DES_74467_01_r2_sliding.txt",
          "values": [-759337, -760461, -761585]
        },
        {
          "fileName": "DES_74467_02_r2_sliding.txt",
          "values": [273.4, 273.2, 272.9]
        }
      ]
    }
  ]
}
```

### Health Check

**GET /health**

Get the current status and health of the service.

## Implementation Details

### Features Used for Classification

The system extracts the following features from discharge data:
- Statistical features (mean, standard deviation, min, max, median)
- Signal correlations for discharges with multiple signals

### Model Training

- XGBoost binary classifier with logistic objective
- Feature scaling using StandardScaler
- Model persistence to disk for later inference
- Customizable hyperparameters via API

### Anomaly Detection

Anomalies are identified based on:
1. The presence of `anomalyTime` in training data (label=1)
2. The absence of `anomalyTime` indicates normal data (label=0)
3. A threshold of 0.5 on the prediction probability

