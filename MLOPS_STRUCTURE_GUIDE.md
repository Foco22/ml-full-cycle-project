# MLOps Structure Guide - model_balanced Deployment

**Document Purpose:** Guide to build a complete MLOps structure for deploying the model_balanced from notebook to production.

**Created:** 2025-12-21
**Model:** model_balanced (XGBoost, 13 features, balanced regularization)
**Current Status:** Development (notebook only)
**Target:** Production-ready deployment with full MLOps

---

## Table of Contents

1. [Current State](#current-state)
2. [What You Need to Build](#what-you-need-to-build)
3. [Detailed Component Specifications](#detailed-component-specifications)
4. [Implementation Priority](#implementation-priority)
5. [File-by-File Guide](#file-by-file-guide)
6. [Configuration Files](#configuration-files)
7. [Testing Strategy](#testing-strategy)
8. [Deployment Options](#deployment-options)
9. [Quick Start Checklist](#quick-start-checklist)

---

## Current State

### What You Have

```
✅ Data Ingestion Pipeline (COMPLETE)
   - src/ingestion/api_data_fetcher.py
   - src/ingestion/bigquery_loader.py
   - src/ingestion/preprocessor.py
   - pipelines/data_ingestion_pipeline.py

✅ Model Development (COMPLETE)
   - notebooks/notebook.ipynb
   - Contains 3 trained models:
     * model (70 features)
     * model_regularized (9 features)
     * model_balanced (13 features) ← TARGET FOR DEPLOYMENT

✅ Utilities (COMPLETE)
   - src/utils/gcs_utils.py
   - src/utils/config_loader.py
   - src/utils/logger.py

✅ Configuration (PARTIAL)
   - config/secrets.yaml
```

### What's Missing

```
❌ Model Training Module (TO BUILD)
❌ Model Inference Module (TO BUILD)
❌ Deployment API (TO BUILD)
❌ Model Monitoring (TO BUILD)
❌ Training Pipeline (TO BUILD)
❌ Tests (TO BUILD)
❌ CI/CD Workflows (TO BUILD)
❌ Configuration Files (TO CREATE)
```

---

## What You Need to Build

### Directory Structure to Create

```
ML - Project/
│
├── src/
│   ├── training/              ← CREATE THIS
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── model_builder.py
│   │   ├── feature_selector.py
│   │   └── model_validator.py
│   │
│   ├── inference/             ← CREATE THIS
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── feature_engineer.py
│   │   └── batch_predict.py
│   │
│   └── monitoring/            ← CREATE THIS
│       ├── __init__.py
│       ├── model_monitor.py
│       ├── data_drift.py
│       └── performance_tracker.py
│
├── pipelines/
│   ├── training_pipeline.py   ← CREATE THIS
│   ├── feature_pipeline.py    ← CREATE THIS
│   └── evaluation_pipeline.py ← CREATE THIS
│
├── deployment/                ← CREATE THIS
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── gunicorn_config.py
│   └── healthcheck.py
│
├── config/
│   ├── model_config.yaml      ← CREATE THIS
│   ├── training_config.yaml   ← CREATE THIS
│   ├── deployment_config.yaml ← CREATE THIS
│   └── feature_config.yaml    ← CREATE THIS
│
├── models/                    ← POPULATE THIS
│   ├── model_balanced_v1.pkl
│   ├── feature_list.json
│   ├── scaler.pkl (if needed)
│   ├── model_metadata.json
│   └── model_config.yaml
│
├── tests/                     ← CREATE THIS
│   ├── test_training.py
│   ├── test_inference.py
│   ├── test_api.py
│   └── fixtures/
│
└── .github/workflows/         ← CREATE THIS
    ├── train_model.yml
    ├── deploy_model.yml
    └── test.yml
```

---

## Detailed Component Specifications

### 1. Model Training Module (src/training/)

#### Purpose
Extract training logic from notebook into reusable, production-ready code.

#### Files to Create

##### `src/training/__init__.py`
```python
"""
Training module for model_balanced.
Handles model training, validation, and artifact generation.
"""

from .train import train_model
from .model_builder import build_xgboost_model
from .feature_selector import select_balanced_features
from .model_validator import validate_model

__all__ = [
    'train_model',
    'build_xgboost_model',
    'select_balanced_features',
    'validate_model'
]
```

##### `src/training/train.py`
Main training orchestrator.

**What it does:**
1. Loads data from BigQuery or local CSV
2. Applies feature selection (13 balanced features)
3. Splits data into train/test
4. Trains XGBoost model
5. Validates performance
6. Saves model artifacts
7. Logs metrics and metadata

**Key functions:**
```python
def load_training_data(config):
    """Load data from BigQuery or CSV"""
    pass

def prepare_features(df, config):
    """Apply feature engineering and selection"""
    pass

def train_model(config_path: str):
    """Main training function"""
    # 1. Load config
    # 2. Load data
    # 3. Prepare features
    # 4. Build model
    # 5. Train
    # 6. Validate
    # 7. Save artifacts
    pass

def save_model_artifacts(model, features, metadata, version):
    """Save model, features, and metadata"""
    pass
```

**Inputs:**
- Configuration file: `config/training_config.yaml`
- Data source: BigQuery `data_ingestion.raw_data` or CSV

**Outputs:**
- `models/model_balanced_v{X}.pkl` - Serialized XGBoost model
- `models/feature_list.json` - List of 13 features used
- `models/model_metadata.json` - Training metrics, timestamp, version
- `logs/training_{timestamp}.log` - Detailed training log

**Example metadata.json:**
```json
{
  "model_name": "model_balanced",
  "version": "v1",
  "created_at": "2025-12-21T10:30:00Z",
  "features": 13,
  "hyperparameters": {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.01,
    "n_estimators": 1000,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0
  },
  "metrics": {
    "train_rmse": 45.2,
    "test_rmse": 52.8,
    "train_r2": 0.92,
    "test_r2": 0.88
  },
  "data": {
    "train_samples": 1200,
    "test_samples": 300,
    "date_range": "2020-01-01 to 2025-12-21"
  }
}
```

##### `src/training/model_builder.py`
XGBoost model configuration.

**What it does:**
- Defines hyperparameters for model_balanced
- Builds XGBoost regressor instance
- Handles model configuration loading

**Key functions:**
```python
def get_balanced_params():
    """Returns hyperparameters for balanced model"""
    return {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'reg_alpha': 0.5,     # L1 regularization
        'reg_lambda': 1.0,    # L2 regularization
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }



def build_xgboost_model(config=None):
    """Build XGBoost model with config"""
    import xgboost as xgb
    params = config if config else get_balanced_params()
    return xgb.XGBRegressor(**params)
```

##### `src/training/feature_selector.py`
Feature selection logic (13 balanced features).

**What it does:**
- Defines which 13 features to use
- Applies feature engineering
- Handles feature transformations

**Key functions:**
```python
def get_balanced_features():
    """Returns list of 13 balanced features"""
    # Extract from notebook
    return [
        'feature1',
        'feature2',
        # ... 13 features total
    ]

def select_balanced_features(X):
    """Select 13 balanced features from dataframe"""
    features = get_balanced_features()
    return X[features]

def engineer_features(df):
    """Apply feature engineering transformations"""
    # Date features, lags, rolling windows, etc.
    pass
```

##### `src/training/model_validator.py`
Model validation and metrics.

**What it does:**
- Calculates performance metrics (RMSE, R2, MAE)
- Compares train vs test performance
- Detects overfitting
- Generates validation reports

**Key functions:**
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }

def validate_model(model, X_train, y_train, X_test, y_test):
    """Validate model performance"""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    return {
        'train': train_metrics,
        'test': test_metrics,
        'overfitting_check': check_overfitting(train_metrics, test_metrics)
    }

def check_overfitting(train_metrics, test_metrics):
    """Detect overfitting"""
    rmse_diff = test_metrics['rmse'] - train_metrics['rmse']
    r2_diff = train_metrics['r2'] - test_metrics['r2']

    return {
        'is_overfitting': rmse_diff > 10 or r2_diff > 0.1,
        'rmse_difference': rmse_diff,
        'r2_difference': r2_diff
    }
```

---

### 2. Model Inference Module (src/inference/)

#### Purpose
Load trained model and generate predictions.

#### Files to Create

##### `src/inference/__init__.py`
```python
"""
Inference module for model_balanced.
Handles model loading, feature engineering, and predictions.
"""

from .predictor import Predictor, predict_single, predict_batch
from .feature_engineer import prepare_inference_features

__all__ = ['Predictor', 'predict_single', 'predict_batch', 'prepare_inference_features']
```

##### `src/inference/predictor.py`
Core prediction logic.

**What it does:**
1. Loads model from pickle file
2. Loads feature list
3. Validates input data
4. Applies feature engineering
5. Generates predictions

**Key class:**
```python
import pickle
import json
import numpy as np
from typing import Dict, List, Union
from .feature_engineer import prepare_inference_features

class Predictor:
    """Model predictor for model_balanced"""

    def __init__(self, model_path: str):
        """
        Initialize predictor

        Args:
            model_path: Path to model .pkl file (e.g., 'models/model_balanced_v1.pkl')
        """
        self.model = self._load_model(model_path)
        self.features = self._load_features(model_path)
        self.metadata = self._load_metadata(model_path)

    def _load_model(self, model_path):
        """Load pickled model"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _load_features(self, model_path):
        """Load feature list"""
        feature_path = model_path.replace('.pkl', '_features.json')
        with open(feature_path, 'r') as f:
            return json.load(f)

    def _load_metadata(self, model_path):
        """Load model metadata"""
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Generate prediction

        Args:
            input_data: Dict or DataFrame with features

        Returns:
            Prediction value(s)
        """
        # 1. Validate input
        self._validate_input(input_data)

        # 2. Prepare features
        X = prepare_inference_features(input_data, self.features)

        # 3. Generate prediction
        prediction = self.model.predict(X)

        return prediction

    def _validate_input(self, input_data):
        """Validate input has required features"""
        if isinstance(input_data, dict):
            missing = set(self.features) - set(input_data.keys())
            if missing:
                raise ValueError(f"Missing features: {missing}")
        elif isinstance(input_data, pd.DataFrame):
            missing = set(self.features) - set(input_data.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")

# Convenience functions
def predict_single(model_path: str, input_data: Dict) -> float:
    """Single prediction"""
    predictor = Predictor(model_path)
    return predictor.predict(input_data)[0]

def predict_batch(model_path: str, input_df: pd.DataFrame) -> np.ndarray:
    """Batch prediction"""
    predictor = Predictor(model_path)
    return predictor.predict(input_df)
```

**Usage example:**
```python
# Single prediction
from src.inference import Predictor

predictor = Predictor('models/model_balanced_v1.pkl')

input_data = {
    'feature1': 100,
    'feature2': 200,
    # ... all 13 features
}

prediction = predictor.predict(input_data)
print(f"Prediction: {prediction[0]}")

# Batch prediction
import pandas as pd
df = pd.read_csv('new_data.csv')
predictions = predictor.predict(df)
```

##### `src/inference/feature_engineer.py`
Feature engineering for inference.

**CRITICAL:** Must match training feature engineering exactly.

**What it does:**
- Applies same transformations as training
- Validates feature values
- Handles missing values

**Key functions:**
```python
import pandas as pd

def prepare_inference_features(input_data, feature_list):
    """
    Prepare features for inference

    Must match training feature engineering EXACTLY
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Apply same transformations as training
    df = engineer_features(df)

    # Select only required features
    df = df[feature_list]

    return df

def engineer_features(df):
    """
    Apply feature engineering

    This must match src/training/feature_selector.py exactly
    """
    # Date features
    # Lag features
    # Rolling windows
    # etc.

    return df
```

##### `src/inference/batch_predict.py`
Batch prediction script.

**What it does:**
- Loads model
- Reads large datasets (CSV, BigQuery)
- Generates predictions in batches
- Saves results

**Key script:**
```python
import argparse
import pandas as pd
from .predictor import Predictor

def batch_predict(
    model_path: str,
    input_path: str,
    output_path: str,
    batch_size: int = 1000
):
    """
    Generate batch predictions

    Args:
        model_path: Path to model file
        input_path: Path to input CSV
        output_path: Path to output CSV
        batch_size: Batch size for processing
    """
    # Load model
    predictor = Predictor(model_path)

    # Read input in chunks
    chunks = []
    for chunk in pd.read_csv(input_path, chunksize=batch_size):
        predictions = predictor.predict(chunk)
        chunk['prediction'] = predictions
        chunks.append(chunk)

    # Combine and save
    result = pd.concat(chunks)
    result.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(f"Total rows: {len(result)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=1000)

    args = parser.parse_args()

    batch_predict(
        args.model,
        args.input,
        args.output,
        args.batch_size
    )
```

**Usage:**
```bash
python -m src.inference.batch_predict \
  --model models/model_balanced_v1.pkl \
  --input data/new_data.csv \
  --output data/predictions.csv \
  --batch-size 5000
```

---

### 3. Deployment API (deployment/)

#### Purpose
Serve model predictions via REST API.

#### Files to Create

##### `deployment/app.py`
FastAPI application.

**What it does:**
- Loads model on startup
- Exposes REST endpoints
- Handles validation
- Logs requests

**Key implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import sys
sys.path.append('..')
from src.inference.predictor import Predictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Exchange Rate Predictor API",
    description="API for model_balanced predictions",
    version="1.0.0"
)

# Load model on startup
MODEL_PATH = "models/model_balanced_v1.pkl"
predictor = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global predictor
    try:
        predictor = Predictor(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Request/Response models
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # ... all 13 features

    class Config:
        schema_extra = {
            "example": {
                "feature1": 100.0,
                "feature2": 200.0,
                # ...
            }
        }

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, float]]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    count: int

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    try:
        input_data = request.dict()
        prediction = predictor.predict(input_data)[0]

        from datetime import datetime
        return PredictionResponse(
            prediction=float(prediction),
            model_version=predictor.metadata['version'],
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    try:
        import pandas as pd
        df = pd.DataFrame(request.data)
        predictions = predictor.predict(df)

        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            model_version=predictor.metadata['version'],
            count=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": predictor.metadata['version'] if predictor else None
    }

@app.get("/model_info")
async def model_info():
    """Model information endpoint"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "metadata": predictor.metadata,
        "features": predictor.features,
        "feature_count": len(predictor.features)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Exchange Rate Predictor API",
        "docs": "/docs",
        "health": "/health"
    }
```

**Test locally:**
```bash
# Install FastAPI
pip install fastapi uvicorn

# Run server
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model_info

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 100, "feature2": 200, ...}'
```

##### `deployment/Dockerfile`
Container definition.

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "deployment.app:app", "-c", "deployment/gunicorn_config.py"]
```

**Build and run:**
```bash
# Build image
docker build -t exchange-rate-predictor .

# Run container
docker run -p 8000:8000 exchange-rate-predictor

# Test
curl http://localhost:8000/health
```

##### `deployment/requirements.txt`
Deployment dependencies.

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
gunicorn==21.2.0
pydantic==2.5.3
xgboost==2.0.3
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.3
```

##### `deployment/gunicorn_config.py`
Production server configuration.

```python
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "logs/api_access.log"
errorlog = "logs/api_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "exchange_rate_predictor"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "path/to/key.pem"
# certfile = "path/to/cert.pem"
```

##### `deployment/healthcheck.py`
Standalone health check script.

```python
#!/usr/bin/env python3
import requests
import sys

def check_health(url="http://localhost:8000/health"):
    """Check API health"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy' and data.get('model_loaded'):
                print("✓ API is healthy")
                return 0
        print("✗ API is unhealthy")
        return 1
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(check_health())
```

---

### 4. Configuration Files (config/)

#### Files to Create

##### `config/model_config.yaml`
Model hyperparameters.

```yaml
# Model configuration for model_balanced

model:
  name: model_balanced
  type: xgboost
  task: regression
  version: v1

hyperparameters:
  objective: reg:squarederror
  max_depth: 6
  learning_rate: 0.01
  n_estimators: 1000

  # Regularization (balanced)
  reg_alpha: 0.5      # L1 regularization
  reg_lambda: 1.0     # L2 regularization

  # Sampling
  subsample: 0.8
  colsample_bytree: 0.8
  colsample_bylevel: 1.0
  colsample_bynode: 1.0

  # Other
  random_state: 42
  n_jobs: -1
  verbosity: 1

features:
  count: 13
  selection_method: balanced_regularization
  # Actual feature names to be extracted from notebook
  names:
    - feature1
    - feature2
    # ... (extract from notebook)

validation:
  test_size: 0.2
  random_state: 42
  shuffle: true

  metrics:
    - rmse
    - r2
    - mae

  thresholds:
    min_r2: 0.7
    max_rmse: 100.0
```

##### `config/training_config.yaml`
Training pipeline settings.

```yaml
# Training configuration

data:
  source: bigquery  # or 'csv'

  bigquery:
    project_id: ${GCP_PROJECT_ID}
    dataset: data_ingestion
    table: raw_data

  csv:
    path: data/training_data.csv

  date_range:
    start: "2020-01-01"
    end: null  # null = today

  target_column: "target_value"

feature_engineering:
  config_path: config/feature_config.yaml

model:
  config_path: config/model_config.yaml

output:
  model_dir: models/
  logs_dir: logs/
  version: auto  # auto-increment or specify manually

  save_format: pickle  # or joblib

  artifacts:
    - model
    - features
    - metadata
    - scaler

training:
  early_stopping:
    enabled: true
    patience: 50
    metric: validation_rmse
    min_delta: 0.01

  cross_validation:
    enabled: false
    folds: 5
    shuffle: true

  verbose: true
  log_interval: 100

validation:
  enabled: true
  check_overfitting: true

  overfitting_thresholds:
    rmse_diff: 10.0
    r2_diff: 0.1

notifications:
  enabled: false
  webhook_url: null
```

##### `config/deployment_config.yaml`
API deployment settings.

```yaml
# Deployment configuration

api:
  title: "Exchange Rate Predictor API"
  description: "API for model_balanced predictions"
  version: "1.0.0"

  server:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    timeout: 120

  cors:
    enabled: true
    allow_origins:
      - "*"
    allow_methods:
      - "GET"
      - "POST"
    allow_headers:
      - "*"

  rate_limiting:
    enabled: true
    requests_per_minute: 60

  authentication:
    enabled: false
    type: api_key
    header_name: X-API-Key

model:
  path: models/model_balanced_v1.pkl
  cache_enabled: true
  reload_on_change: false

  prediction:
    batch_size_limit: 1000
    timeout: 30

logging:
  level: INFO
  format: json

  files:
    access: logs/api_access.log
    error: logs/api_error.log
    application: logs/api.log

  rotation:
    max_bytes: 10485760  # 10MB
    backup_count: 5

monitoring:
  enabled: false
  prometheus:
    enabled: false
    port: 9090
```

##### `config/feature_config.yaml`
Feature engineering rules.

```yaml
# Feature engineering configuration

features:
  # Extract from notebook - 13 balanced features
  selected_features:
    - feature1
    - feature2
    - feature3
    # ... (total 13 features)

engineering:
  date_features:
    enabled: true
    extract:
      - year
      - month
      - day
      - dayofweek

  lag_features:
    enabled: true
    lags:
      - 1
      - 7
      - 30
    columns:
      - usdclp_obs
      - eurclp_obs

  rolling_features:
    enabled: true
    windows:
      - 7
      - 30
    functions:
      - mean
      - std
    columns:
      - usdclp_obs
      - eurclp_obs

  scaling:
    enabled: false
    method: standard  # standard, minmax, robust
    per_feature: false

  encoding:
    categorical_method: onehot
    handle_unknown: ignore

validation:
  check_missing: true
  missing_threshold: 0.1

  check_outliers: true
  outlier_method: iqr
  outlier_threshold: 3.0

  check_dtypes: true
```

---

### 5. Pipelines (pipelines/)

#### Files to Create

##### `pipelines/training_pipeline.py`
Complete training orchestration.

```python
"""
Training pipeline for model_balanced

Orchestrates:
1. Data ingestion
2. Feature engineering
3. Model training
4. Validation
5. Model registration
6. Evaluation
"""

import argparse
import yaml
from pathlib import Path
import sys
sys.path.append('..')

from src.training.train import train_model
from src.utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

def run_training_pipeline(config_path: str):
    """
    Execute complete training pipeline

    Args:
        config_path: Path to training configuration YAML
    """
    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")

        # 2. Data ingestion (optional - if needed)
        logger.info("Step 1: Data ingestion...")
        # If data needs to be fetched
        # run_data_ingestion()

        # 3. Feature engineering
        logger.info("Step 2: Feature engineering...")
        # from pipelines.feature_pipeline import run_feature_pipeline
        # run_feature_pipeline(config)

        # 4. Model training
        logger.info("Step 3: Model training...")
        model_path, metrics = train_model(config)
        logger.info(f"Model trained successfully: {model_path}")
        logger.info(f"Metrics: {metrics}")

        # 5. Validation
        logger.info("Step 4: Model validation...")
        # Additional validation if needed

        # 6. Model registration
        logger.info("Step 5: Model registration...")
        # Register model version in tracking system

        # 7. Generate report
        logger.info("Step 6: Generating training report...")
        generate_training_report(model_path, metrics, config)

        # Success
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 80)
        logger.info(f"TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Model: {model_path}")
        logger.info("=" * 80)

        return model_path, metrics

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise

def generate_training_report(model_path, metrics, config):
    """Generate training report"""
    report_path = Path('logs') / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, 'w') as f:
        f.write("TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {model_path}\n\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nConfiguration:\n")
        f.write(yaml.dump(config, indent=2))

    logger.info(f"Training report saved to {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to training configuration file'
    )

    args = parser.parse_args()

    run_training_pipeline(args.config)
```

**Usage:**
```bash
python pipelines/training_pipeline.py --config config/training_config.yaml
```

---

## Implementation Priority

### Phase 1: Core Training (HIGH PRIORITY)
**Goal:** Extract model from notebook and make it reproducible

**Tasks:**
1. Create `src/training/` module
   - `train.py` - main training script
   - `model_builder.py` - XGBoost configuration
   - `feature_selector.py` - 13 feature selection
   - `model_validator.py` - metrics calculation

2. Create `config/model_config.yaml`
   - Extract hyperparameters from notebook
   - Document 13 features

3. Run training and save artifacts
   - `models/model_balanced_v1.pkl`
   - `models/feature_list.json`
   - `models/model_metadata.json`

**Validation:**
- Model trains successfully
- Metrics match notebook
- Artifacts saved correctly

**Estimated time:** 1-2 days

---

### Phase 2: Inference (HIGH PRIORITY)
**Goal:** Load model and generate predictions

**Tasks:**
1. Create `src/inference/` module
   - `predictor.py` - prediction class
   - `feature_engineer.py` - feature prep
   - `batch_predict.py` - batch script

2. Test predictions
   - Load saved model
   - Predict on test data
   - Verify results match notebook

**Validation:**
- Model loads correctly
- Predictions work
- Features align with training

**Estimated time:** 1 day

---

### Phase 3: Deployment API (MEDIUM PRIORITY)
**Goal:** Serve predictions via REST API

**Tasks:**
1. Create `deployment/app.py`
   - FastAPI application
   - `/predict` endpoint
   - `/health` endpoint
   - `/model_info` endpoint

2. Create `deployment/Dockerfile`

3. Test locally
   - Run with uvicorn
   - Test endpoints
   - Validate responses

**Validation:**
- API starts successfully
- Endpoints respond correctly
- Docker container works

**Estimated time:** 1-2 days

---

### Phase 4: Pipelines (MEDIUM PRIORITY)
**Goal:** Orchestrate end-to-end workflows

**Tasks:**
1. Create `pipelines/training_pipeline.py`
2. Create `pipelines/feature_pipeline.py`
3. Create configuration files

**Validation:**
- Pipelines run end-to-end
- Logging works
- Artifacts generated

**Estimated time:** 1 day

---

### Phase 5: Monitoring (LOW PRIORITY)
**Goal:** Track model performance

**Tasks:**
1. Create `src/monitoring/` module
2. Implement drift detection
3. Performance tracking

**Estimated time:** 2-3 days

---

### Phase 6: Testing & CI/CD (LOW PRIORITY)
**Goal:** Automated testing and deployment

**Tasks:**
1. Create test suite
2. Setup GitHub Actions
3. Automated deployment

**Estimated time:** 2-3 days

---

## Quick Start Checklist

### Week 1: Extract Model from Notebook

- [ ] Create `src/training/` directory structure
- [ ] Extract hyperparameters from notebook → `config/model_config.yaml`
- [ ] Extract 13 feature names from notebook → `config/feature_config.yaml`
- [ ] Write `src/training/train.py` - main training logic
- [ ] Write `src/training/model_builder.py` - XGBoost builder
- [ ] Write `src/training/feature_selector.py` - feature selection
- [ ] Write `src/training/model_validator.py` - metrics calculation
- [ ] Run training and save model to `models/model_balanced_v1.pkl`
- [ ] Save feature list to `models/feature_list.json`
- [ ] Save metadata to `models/model_metadata.json`
- [ ] Verify metrics match notebook

### Week 2: Build Inference

- [ ] Create `src/inference/` directory structure
- [ ] Write `src/inference/predictor.py` - Predictor class
- [ ] Write `src/inference/feature_engineer.py` - feature prep
- [ ] Write `src/inference/batch_predict.py` - batch script
- [ ] Test single prediction
- [ ] Test batch prediction
- [ ] Verify predictions match notebook

### Week 3: Build API

- [ ] Create `deployment/` directory
- [ ] Write `deployment/app.py` - FastAPI application
- [ ] Write `deployment/Dockerfile`
- [ ] Create `deployment/requirements.txt`
- [ ] Write `deployment/gunicorn_config.py`
- [ ] Test API locally with uvicorn
- [ ] Test `/predict` endpoint
- [ ] Test `/health` endpoint
- [ ] Build Docker image
- [ ] Test Docker container

### Week 4: Polish & Deploy

- [ ] Create `pipelines/training_pipeline.py`
- [ ] Write tests for critical components
- [ ] Document API with examples
- [ ] Deploy to GCP Cloud Run (or preferred platform)
- [ ] Setup monitoring (basic)
- [ ] Create deployment documentation

---

## Next Steps

1. **Start with Phase 1** - Extract training code from notebook
2. **Test thoroughly** - Ensure metrics match notebook
3. **Build incrementally** - One phase at a time
4. **Document as you go** - Update this guide with actual implementations
5. **Deploy early** - Get API running as soon as Phase 2 is complete

---

## Questions to Answer from Notebook

Before building, extract these from `notebooks/notebook.ipynb`:

1. **What are the 13 balanced features?**
   - Extract feature names
   - Document in `config/feature_config.yaml`

2. **What are the exact hyperparameters?**
   - Extract `params_balanced`
   - Document in `config/model_config.yaml`

3. **What feature engineering is applied?**
   - Date features?
   - Lag features?
   - Rolling windows?
   - Document in feature engineering code

4. **What are the performance metrics?**
   - Train RMSE, R2
   - Test RMSE, R2
   - Document as baseline

5. **What scaling/normalization is used?**
   - StandardScaler?
   - MinMaxScaler?
   - None?

---

**Document Status:** Living document - update as you build
**Last Updated:** 2025-12-21
**Next Review:** After Phase 1 completion
