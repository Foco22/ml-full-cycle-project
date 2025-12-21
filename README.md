# ML Full-Cycle Project - Exchange Rate Prediction

[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Enabled-blue)](https://github.com/Foco22/ml-full-cycle-project/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![BigQuery](https://img.shields.io/badge/BigQuery-Enabled-orange)](https://cloud.google.com/bigquery)
[![MLOps](https://img.shields.io/badge/MLOps-Ready-green)](https://ml-ops.org/)

End-to-end ML project that predicts exchange rates (USD/CLP, EUR/CLP, UF) using XGBoost. Includes data ingestion, model training, deployment, and monitoring.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [MLOps Architecture](#mlops-architecture)
- [Components Guide](#components-guide)
- [Development Roadmap](#development-roadmap)
- [Model Information](#model-information)
- [Deployment Guide](#deployment-guide)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Contributing](#contributing)

---

## Overview

### What This Project Does

```
CMF API → Data Ingestion → Feature Engineering → Model Training → Model Registry
                ↓                                        ↓
          BigQuery Storage                        Deployment API
                                                         ↓
                                                  Predictions + Monitoring
```

### Key Features

- **Data Pipeline**: Automated daily ingestion from CMF Chile API
- **ML Model**: XGBoost regressor with 13 balanced features
- **Deployment**: REST API for predictions (planned)
- **Monitoring**: Model performance tracking (planned)
- **MLOps**: Full CI/CD pipeline (planned)

---

## Quick Start

### 1. Configure GCP (5 min)

```bash
./setup_github_actions.sh
```

### 2. Add GitHub Secrets

Go to: **Settings → Secrets and variables → Actions**

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_CREDENTIALS` | JSON from `gcp-key.json` |
| `CMF_API_KEY` | Get from https://api.cmfchile.cl/ |

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Data Ingestion

```bash
python pipelines/data_ingestion_pipeline.py --source api --mode incremental
```

---

## Project Structure

```
ML - Project/
│
├── notebooks/                      # Development & experimentation
│   └── notebook.ipynb              # Model development (model_balanced)
│
├── src/                            # Source code
│   ├── ingestion/                  # ✅ Data ingestion (COMPLETE)
│   │   ├── api_data_fetcher.py
│   │   ├── bigquery_loader.py
│   │   └── preprocessor.py
│   │
│   ├── training/                   # ⏳ Model training (TO BUILD)
│   │   ├── train.py                # Main training script
│   │   ├── model_builder.py        # XGBoost configuration
│   │   ├── feature_selector.py     # Feature selection logic
│   │   └── model_validator.py      # Model validation
│   │
│   ├── inference/                  # ⏳ Prediction logic (TO BUILD)
│   │   ├── predictor.py            # Model loading & prediction
│   │   ├── feature_engineer.py     # Feature engineering for inference
│   │   └── batch_predict.py        # Batch predictions
│   │
│   ├── monitoring/                 # ⏳ Model monitoring (TO BUILD)
│   │   ├── model_monitor.py        # Track predictions & performance
│   │   ├── data_drift.py           # Detect feature drift
│   │   └── performance_tracker.py  # RMSE, R2 tracking
│   │
│   └── utils/                      # ✅ Utilities (COMPLETE)
│       ├── gcs_utils.py
│       ├── config_loader.py
│       └── logger.py
│
├── pipelines/                      # Orchestration pipelines
│   ├── data_ingestion_pipeline.py  # ✅ Data ingestion (COMPLETE)
│   ├── training_pipeline.py        # ⏳ Full training workflow (TO BUILD)
│   ├── feature_pipeline.py         # ⏳ Feature engineering (TO BUILD)
│   └── evaluation_pipeline.py      # ⏳ Model evaluation (TO BUILD)
│
├── models/                         # Model artifacts
│   ├── model_balanced_v1.pkl       # ⏳ Trained model (TO SAVE)
│   ├── feature_list.json           # ⏳ List of 13 features (TO CREATE)
│   ├── scaler.pkl                  # ⏳ Feature scaler (TO SAVE)
│   ├── model_metadata.json         # ⏳ Version, metrics, timestamp (TO CREATE)
│   └── model_config.yaml           # ⏳ Model hyperparameters (TO CREATE)
│
├── deployment/                     # Deployment artifacts
│   ├── app.py                      # ⏳ FastAPI application (TO BUILD)
│   ├── Dockerfile                  # ⏳ Container definition (TO CREATE)
│   ├── requirements.txt            # ⏳ Deployment dependencies (TO CREATE)
│   ├── gunicorn_config.py          # ⏳ Production server config (TO CREATE)
│   └── healthcheck.py              # ⏳ Health endpoint (TO BUILD)
│
├── config/                         # Configuration files
│   ├── secrets.yaml                # ✅ API keys & credentials (COMPLETE)
│   ├── model_config.yaml           # ⏳ Model hyperparameters (TO CREATE)
│   ├── training_config.yaml        # ⏳ Training settings (TO CREATE)
│   ├── deployment_config.yaml      # ⏳ API settings (TO CREATE)
│   └── feature_config.yaml         # ⏳ Feature engineering rules (TO CREATE)
│
├── tests/                          # Unit & integration tests
│   ├── test_training.py            # ⏳ Test training pipeline (TO BUILD)
│   ├── test_inference.py           # ⏳ Test predictions (TO BUILD)
│   ├── test_api.py                 # ⏳ Test deployment API (TO BUILD)
│   ├── test_features.py            # ⏳ Test feature engineering (TO BUILD)
│   └── fixtures/                   # ⏳ Test data (TO CREATE)
│
├── .github/workflows/              # CI/CD workflows
│   ├── train_model.yml             # ⏳ Automated training (TO CREATE)
│   ├── deploy_model.yml            # ⏳ Automated deployment (TO CREATE)
│   └── test.yml                    # ⏳ Run tests (TO CREATE)
│
├── logs/                           # Application logs
└── requirements.txt                # Python dependencies
```

**Legend:**
- ✅ **COMPLETE** - Already implemented and working
- ⏳ **TO BUILD** - Needs to be created for full MLOps

---

## MLOps Architecture

### Current State (Phase 1 - Data Ingestion)

```
┌─────────────┐
│  CMF API    │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Data Ingestion   │  ✅ COMPLETE
│ Pipeline         │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│   BigQuery      │  ✅ COMPLETE
└─────────────────┘
```

### Target State (Phase 2-4 - Full MLOps)

```
┌─────────────┐
│  CMF API    │
└──────┬──────┘
       │
       ▼
┌──────────────────┐     ┌────────────────┐
│ Data Ingestion   │────▶│  Feature Eng.  │  ⏳ TO BUILD
│ Pipeline         │     │  Pipeline      │
└──────────────────┘     └────────┬───────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │  Training      │  ⏳ TO BUILD
                         │  Pipeline      │
                         └────────┬───────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │ Model Registry │  ⏳ TO BUILD
                         │ (models/)      │
                         └────────┬───────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
       ┌────────────────┐                  ┌────────────────┐
       │  Deployment    │  ⏳ TO BUILD     │  Monitoring    │  ⏳ TO BUILD
       │  API (FastAPI) │                  │  Dashboard     │
       └────────────────┘                  └────────────────┘
```

---

## Components Guide

### 1. Data Ingestion (✅ COMPLETE)

**Location:** `src/ingestion/`, `pipelines/data_ingestion_pipeline.py`

**What it does:**
- Fetches USD/CLP, EUR/CLP, UF data from CMF Chile API
- Loads data into BigQuery
- Runs daily via GitHub Actions
- Handles incremental updates (no duplicates)

**How to use:**
```bash
# Run full ingestion
python pipelines/data_ingestion_pipeline.py --source api --mode full

# Run incremental (default)
python pipelines/data_ingestion_pipeline.py --source api --mode incremental
```

**Key files:**
- `src/ingestion/api_data_fetcher.py` - API client
- `src/ingestion/bigquery_loader.py` - BigQuery operations
- `src/ingestion/preprocessor.py` - Data preprocessing

---

### 2. Model Training (⏳ TO BUILD)

**Location:** `src/training/`, `pipelines/training_pipeline.py`

**What it will do:**
- Extract training logic from `notebooks/notebook.ipynb`
- Train XGBoost model with 13 balanced features
- Validate model performance
- Save model artifacts to `models/`

**How to use (planned):**
```bash
# Train new model
python pipelines/training_pipeline.py --config config/training_config.yaml

# Train with custom parameters
python src/training/train.py --features 13 --regularization balanced
```

**Key files to create:**

#### `src/training/train.py`
Main training script that orchestrates the entire training process.

**Responsibilities:**
```python
# Pseudo-code structure
1. Load data from BigQuery/local
2. Apply feature selection (13 balanced features)
3. Split train/test data
4. Train model_balanced (XGBoost)
5. Validate model (RMSE, R2)
6. Save model to models/model_balanced_v{version}.pkl
7. Save metadata (metrics, features, timestamp)
8. Log results
```

**Expected inputs:**
- Training data (from BigQuery or CSV)
- Configuration file (`config/training_config.yaml`)

**Expected outputs:**
- `models/model_balanced_v{X}.pkl` - Trained model
- `models/feature_list.json` - List of 13 features used
- `models/model_metadata.json` - Metrics, version, timestamp
- `logs/training_{timestamp}.log` - Training logs

#### `src/training/model_builder.py`
XGBoost model configuration and initialization.

**Responsibilities:**
```python
# Contains model hyperparameters
params_balanced = {
    'objective': 'reg:squarederror',
    'max_depth': ...,
    'learning_rate': ...,
    'n_estimators': ...,
    'reg_alpha': ...,  # L1 regularization
    'reg_lambda': ..., # L2 regularization
}

# Function to build model
def build_model(config):
    return xgb.XGBRegressor(**config)
```

#### `src/training/feature_selector.py`
Feature selection logic for the 13 balanced features.

**Responsibilities:**
- Define which 13 features to use
- Apply feature engineering transformations
- Handle feature scaling if needed

#### `src/training/model_validator.py`
Model validation and performance metrics.

**Responsibilities:**
- Calculate RMSE, R2, MAE
- Compare train vs test performance
- Detect overfitting
- Generate validation reports

---

### 3. Model Inference (⏳ TO BUILD)

**Location:** `src/inference/`

**What it will do:**
- Load trained model from `models/`
- Apply same feature engineering as training
- Generate predictions for new data
- Support batch and single predictions

**How to use (planned):**
```bash
# Single prediction
python src/inference/predictor.py --input '{"feature1": 100, "feature2": 200, ...}'

# Batch prediction
python src/inference/batch_predict.py --input-file data/new_data.csv --output-file predictions.csv
```

**Key files to create:**

#### `src/inference/predictor.py`
Core prediction logic.

```python
# Pseudo-code structure
class Predictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.features = load_feature_list()
        self.scaler = load_scaler()

    def predict(self, input_data):
        # 1. Validate input has required features
        # 2. Apply feature engineering
        # 3. Scale features
        # 4. Generate prediction
        # 5. Return result
        return prediction
```

#### `src/inference/feature_engineer.py`
Feature engineering for inference (must match training).

**Critical:** Must apply exact same transformations as training.

#### `src/inference/batch_predict.py`
Batch prediction script for large datasets.

```python
# Pseudo-code
1. Load model using Predictor
2. Read input CSV/BigQuery
3. Apply predictions in batches
4. Save results
5. Generate summary statistics
```

---

### 4. Deployment API (⏳ TO BUILD)

**Location:** `deployment/`

**What it will do:**
- Serve model predictions via REST API
- Handle authentication
- Log requests
- Health checks

**How to use (planned):**
```bash
# Local development
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000

# Production (Docker)
docker build -t exchange-rate-predictor .
docker run -p 8000:8000 exchange-rate-predictor
```

**API Endpoints (planned):**
```
POST   /predict              - Single prediction
POST   /batch_predict        - Batch predictions
GET    /health               - Health check
GET    /model_info           - Model metadata
GET    /metrics              - Model performance metrics
```

**Key files to create:**

#### `deployment/app.py`
FastAPI application.

```python
# Pseudo-code structure
from fastapi import FastAPI
from src.inference.predictor import Predictor

app = FastAPI()
predictor = Predictor("models/model_balanced_v1.pkl")

@app.post("/predict")
async def predict(data: PredictionRequest):
    # Validate input
    # Generate prediction
    # Return result
    return {"prediction": result}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/model_info")
async def model_info():
    # Return model metadata
    return metadata
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 100,
    "feature2": 200,
    ...
  }'
```

**Example response:**
```json
{
  "prediction": 850.5,
  "model_version": "v1",
  "timestamp": "2025-12-21T10:30:00Z",
  "confidence": 0.95
}
```

#### `deployment/Dockerfile`
Container definition for deployment.

```dockerfile
# Pseudo-code structure
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "deployment.app:app", "-c", "deployment/gunicorn_config.py"]
```

#### `deployment/gunicorn_config.py`
Production server configuration.

```python
# Worker configuration
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
timeout = 120
```

---

### 5. Model Monitoring (⏳ TO BUILD)

**Location:** `src/monitoring/`

**What it will do:**
- Track prediction performance over time
- Detect data drift
- Alert on performance degradation
- Generate monitoring dashboards

**How to use (planned):**
```bash
# Run monitoring checks
python src/monitoring/model_monitor.py --check-drift --check-performance

# Generate monitoring report
python src/monitoring/performance_tracker.py --report --last-7-days
```

**Key files to create:**

#### `src/monitoring/model_monitor.py`
Main monitoring orchestrator.

```python
# Monitors:
- Prediction volume
- Prediction distribution
- Response times
- Error rates
- Model performance metrics
```

#### `src/monitoring/data_drift.py`
Feature drift detection.

```python
# Detects:
- Feature distribution changes
- New categorical values
- Missing features
- Outliers
```

#### `src/monitoring/performance_tracker.py`
Performance tracking over time.

```python
# Tracks:
- RMSE over time
- R2 over time
- Prediction accuracy
- Comparison with actual values
```

---

### 6. Pipelines (Orchestration)

**Location:** `pipelines/`

**What it will do:**
- Orchestrate end-to-end workflows
- Handle dependencies between steps
- Enable scheduling and automation

**Key files to create:**

#### `pipelines/training_pipeline.py`
Complete training workflow.

```python
# Pipeline steps:
1. Trigger data ingestion
2. Run feature engineering
3. Train model
4. Validate model
5. Register model (save artifacts)
6. Run evaluation
7. Generate training report
8. Send notifications (optional)
```

**Usage:**
```bash
python pipelines/training_pipeline.py --config config/training_config.yaml
```

#### `pipelines/feature_pipeline.py`
Feature engineering workflow.

```python
# Pipeline steps:
1. Load raw data from BigQuery
2. Apply feature engineering
3. Feature selection (13 features)
4. Save processed features
```

#### `pipelines/evaluation_pipeline.py`
Model evaluation workflow.

```python
# Pipeline steps:
1. Load test data
2. Load model
3. Generate predictions
4. Calculate metrics
5. Compare with previous model
6. Generate evaluation report
```

---

### 7. Configuration Management

**Location:** `config/`

**Key files to create:**

#### `config/model_config.yaml`
Model hyperparameters and settings.

```yaml
model:
  type: xgboost
  task: regression

hyperparameters:
  objective: reg:squarederror
  max_depth: 6
  learning_rate: 0.01
  n_estimators: 1000
  reg_alpha: 0.5    # L1 regularization
  reg_lambda: 1.0   # L2 regularization
  subsample: 0.8
  colsample_bytree: 0.8

features:
  count: 13
  selection_method: balanced_regularization

validation:
  test_size: 0.2
  random_state: 42
  metrics:
    - rmse
    - r2
    - mae
```

#### `config/training_config.yaml`
Training pipeline settings.

```yaml
training:
  data_source: bigquery
  dataset: data_ingestion.raw_data
  date_range:
    start: 2020-01-01
    end: null  # null = today

  output:
    model_dir: models/
    logs_dir: logs/
    version: auto  # auto-increment version

  early_stopping:
    enabled: true
    patience: 50
    metric: validation_rmse

  cross_validation:
    enabled: false
    folds: 5
```

#### `config/deployment_config.yaml`
API deployment settings.

```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 120

  cors:
    enabled: true
    origins:
      - "*"

  rate_limiting:
    enabled: true
    requests_per_minute: 60

  authentication:
    enabled: false
    type: api_key

logging:
  level: INFO
  format: json
  destination: logs/api.log

model:
  path: models/model_balanced_v1.pkl
  cache_enabled: true
  reload_on_change: true
```

#### `config/feature_config.yaml`
Feature engineering rules.

```yaml
features:
  selected_features:
    - feature1
    - feature2
    # ... (13 features total)

  engineering:
    scaling:
      method: standard  # or minmax, robust
      per_feature: false

    encoding:
      categorical_method: onehot

    imputation:
      strategy: mean
      fill_value: null

  validation:
    check_missing: true
    check_outliers: true
    outlier_method: iqr
    outlier_threshold: 3
```

---

### 8. Testing

**Location:** `tests/`

**Key files to create:**

#### `tests/test_training.py`
Test training pipeline.

```python
# Test cases:
- Test data loading
- Test feature selection
- Test model training
- Test model saving
- Test metadata generation
- Test error handling
```

#### `tests/test_inference.py`
Test prediction logic.

```python
# Test cases:
- Test model loading
- Test single prediction
- Test batch prediction
- Test input validation
- Test feature engineering
- Test edge cases
```

#### `tests/test_api.py`
Test deployment API.

```python
# Test cases:
- Test /predict endpoint
- Test /batch_predict endpoint
- Test /health endpoint
- Test /model_info endpoint
- Test authentication
- Test error responses
- Test rate limiting
```

**How to run:**
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_training.py

# Run with coverage
pytest --cov=src tests/
```

---

### 9. CI/CD Workflows

**Location:** `.github/workflows/`

**Key files to create:**

#### `.github/workflows/train_model.yml`
Automated model training workflow.

```yaml
# Triggers:
- Manual dispatch
- Weekly schedule
- On push to main (if training code changes)

# Steps:
1. Setup Python environment
2. Install dependencies
3. Configure GCP credentials
4. Run training pipeline
5. Validate model performance
6. Save model artifacts
7. Create model version tag
8. Send notification
```

#### `.github/workflows/deploy_model.yml`
Automated deployment workflow.

```yaml
# Triggers:
- New model version tag
- Manual dispatch

# Steps:
1. Build Docker image
2. Run tests
3. Push to container registry
4. Deploy to production (GCP Cloud Run / K8s)
5. Run smoke tests
6. Send notification
```

#### `.github/workflows/test.yml`
Automated testing workflow.

```yaml
# Triggers:
- On every push
- On pull requests

# Steps:
1. Setup Python environment
2. Install dependencies
3. Run linting (flake8, black)
4. Run unit tests
5. Run integration tests
6. Generate coverage report
7. Upload coverage to Codecov
```

---

## Development Roadmap

### Phase 1: Data Ingestion (✅ COMPLETE)
- [x] CMF API integration
- [x] BigQuery loader
- [x] GitHub Actions automation
- [x] Incremental updates

### Phase 2: Model Training (⏳ NEXT)
**Priority: HIGH**

1. **Extract training code from notebook**
   - Create `src/training/train.py`
   - Extract feature selection logic
   - Extract model configuration

2. **Create model artifacts**
   - Save `model_balanced.pkl`
   - Create `feature_list.json` (13 features)
   - Generate `model_metadata.json`

3. **Build training pipeline**
   - Create `pipelines/training_pipeline.py`
   - Add configuration files
   - Add logging

4. **Add validation**
   - Create `src/training/model_validator.py`
   - Add train/test metrics
   - Add overfitting checks

**Estimated effort:** 3-5 days

### Phase 3: Model Inference (⏳ PLANNED)
**Priority: HIGH**

1. **Build predictor**
   - Create `src/inference/predictor.py`
   - Load model and features
   - Handle predictions

2. **Feature engineering**
   - Create `src/inference/feature_engineer.py`
   - Match training transformations
   - Add validation

3. **Batch predictions**
   - Create `src/inference/batch_predict.py`
   - Handle large datasets
   - Optimize performance

**Estimated effort:** 2-3 days

### Phase 4: Deployment API (⏳ PLANNED)
**Priority: MEDIUM**

1. **Build FastAPI app**
   - Create `deployment/app.py`
   - Add /predict endpoint
   - Add /health endpoint
   - Add /model_info endpoint

2. **Containerize**
   - Create `Dockerfile`
   - Create `deployment/requirements.txt`
   - Test locally

3. **Deploy**
   - Deploy to GCP Cloud Run
   - Setup CI/CD
   - Add monitoring

**Estimated effort:** 3-4 days

### Phase 5: Monitoring (⏳ PLANNED)
**Priority: MEDIUM**

1. **Model monitoring**
   - Create `src/monitoring/model_monitor.py`
   - Track predictions
   - Track performance

2. **Data drift detection**
   - Create `src/monitoring/data_drift.py`
   - Monitor feature distributions
   - Alert on drift

3. **Performance tracking**
   - Create `src/monitoring/performance_tracker.py`
   - RMSE/R2 over time
   - Comparison with actuals

**Estimated effort:** 3-4 days

### Phase 6: Testing & CI/CD (⏳ PLANNED)
**Priority: LOW**

1. **Unit tests**
   - Training tests
   - Inference tests
   - API tests

2. **CI/CD workflows**
   - Automated training
   - Automated deployment
   - Automated testing

**Estimated effort:** 2-3 days

---

## Model Information

### Current Model: model_balanced

**Type:** XGBoost Regressor
**Task:** Exchange rate prediction
**Features:** 13 balanced features (regularization)
**Location:** `notebooks/notebook.ipynb` (to be exported)

### Model Versions

| Version | Features | Regularization | Status | Location |
|---------|----------|----------------|--------|----------|
| **model** | 70 | None | Development | notebook |
| **model_regularized** | 9 | Aggressive | Development | notebook |
| **model_balanced** | 13 | Balanced | **Production Ready** | notebook |

### Performance Metrics (from notebook)

```
Model: model_balanced (13 features)
- Train RMSE: [to be extracted]
- Test RMSE: [to be extracted]
- Train R²: [to be extracted]
- Test R²: [to be extracted]
```

### Features (13 balanced features)

To be documented after extraction from notebook.

### Hyperparameters

```python
params_balanced = {
    'objective': 'reg:squarederror',
    # Full parameters to be extracted from notebook
}
```

---

## Deployment Guide

### Local Development

```bash
# 1. Clone repository
git clone <repo-url>
cd ML\ -\ Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env with your credentials

# 4. Run data ingestion
python pipelines/data_ingestion_pipeline.py

# 5. Train model (once implemented)
python pipelines/training_pipeline.py

# 6. Start API (once implemented)
uvicorn deployment.app:app --reload
```

### Production Deployment (Planned)

#### Option 1: GCP Cloud Run (Recommended)

```bash
# 1. Build Docker image
docker build -t gcr.io/YOUR_PROJECT/exchange-rate-predictor .

# 2. Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT/exchange-rate-predictor

# 3. Deploy to Cloud Run
gcloud run deploy exchange-rate-predictor \
  --image gcr.io/YOUR_PROJECT/exchange-rate-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Option 2: Kubernetes

```bash
# 1. Create deployment
kubectl apply -f k8s/deployment.yaml

# 2. Expose service
kubectl apply -f k8s/service.yaml

# 3. Check status
kubectl get pods
kubectl get services
```

#### Option 3: Docker Compose (Local/Testing)

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Monitoring & Maintenance

### Model Performance Monitoring (Planned)

**Metrics to track:**
- RMSE over time
- R² score over time
- Prediction distribution
- Actual vs Predicted comparison
- Error analysis

**Alerts:**
- Performance degradation (RMSE increase > 10%)
- Data drift detection
- API errors > threshold
- High latency (> 500ms)

### Model Retraining Strategy (Planned)

**Triggers for retraining:**
1. **Scheduled:** Weekly/Monthly retraining
2. **Performance-based:** RMSE degradation > threshold
3. **Data drift:** Feature distribution changes
4. **Manual:** On-demand retraining

**Retraining process:**
```
1. Trigger detected
2. Pull latest data from BigQuery
3. Run training pipeline
4. Validate new model performance
5. Compare with current model
6. If better: Deploy new model
7. If worse: Alert and investigate
```

### Data Quality Checks (Planned)

- Missing values validation
- Outlier detection
- Feature range validation
- Data freshness checks
- Schema validation

---

## Tech Stack

### Current
- **Language:** Python 3.10
- **ML Framework:** XGBoost, Scikit-learn
- **Cloud:** Google Cloud Platform (BigQuery)
- **CI/CD:** GitHub Actions
- **Data Source:** CMF Chile API

### Planned
- **API Framework:** FastAPI
- **Containerization:** Docker
- **Orchestration:** Airflow / Prefect (optional)
- **Monitoring:** Prometheus + Grafana (optional)
- **Model Registry:** MLflow (optional)

---

## Data Schema

### BigQuery Table: `data_ingestion.raw_data`

| Column | Type | Description |
|--------|------|-------------|
| Fecha | DATE | Exchange rate date |
| usdclp_obs | FLOAT64 | USD to CLP rate |
| eurclp_obs | FLOAT64 | EUR to CLP rate |
| ufclp | FLOAT64 | UF value in CLP |
| ingestion_timestamp | TIMESTAMP | Ingestion time |
| data_source | STRING | "CMF_Chile_API" |

### Model Input Schema (Planned)

```json
{
  "feature1": float,
  "feature2": float,
  ...
  "feature13": float
}
```

### Model Output Schema (Planned)

```json
{
  "prediction": float,
  "model_version": string,
  "timestamp": string,
  "confidence": float
}
```

---

## Cost Estimation

### Current Costs
- **BigQuery Storage:** ~$0.02/month
- **BigQuery Queries:** Free (1 TB/month)
- **GitHub Actions:** Free (2,000 min/month)

**Total:** ~$0.10 USD/month

### Projected Costs (with deployment)
- **Cloud Run:** ~$5-10/month (with free tier)
- **Container Registry:** ~$0.50/month
- **Monitoring:** Free (Google Cloud Operations)

**Total:** ~$5-15 USD/month

---

## Troubleshooting

### Common Issues

#### 1. Model not loading
```bash
# Check model file exists
ls -la models/model_balanced_*.pkl

# Check file permissions
chmod 644 models/model_balanced_*.pkl
```

#### 2. Feature mismatch
```bash
# Verify feature list matches
python -c "import json; print(json.load(open('models/feature_list.json')))"
```

#### 3. API not starting
```bash
# Check logs
tail -f logs/api.log

# Test locally
curl http://localhost:8000/health
```

#### 4. BigQuery connection issues
```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS="config/gcp-key.json"

# Test connection
python -c "from google.cloud import bigquery; client = bigquery.Client(); print('Connected!')"
```

---

## Best Practices

### Code Quality
- Use type hints
- Write docstrings
- Follow PEP 8
- Run linting (flake8, black)

### Version Control
- Use semantic versioning for models
- Tag releases
- Write meaningful commit messages
- Use feature branches

### Security
- Never commit secrets
- Use environment variables
- Rotate API keys regularly
- Use least privilege GCP permissions

### Testing
- Write unit tests for all modules
- Integration tests for pipelines
- API tests for endpoints
- Achieve > 80% code coverage

### Documentation
- Keep README updated
- Document all APIs
- Add inline comments for complex logic
- Update changelog

---

## Contributing

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Review Process

1. All PRs require review
2. All tests must pass
3. Code coverage must not decrease
4. Follow style guide

---

## License

Open source for educational purposes.

---

## Contact & Support

- **Issues:** Open an issue in this repository
- **Discussions:** Use GitHub Discussions
- **Security:** Report security issues privately

---

## Acknowledgments

- CMF Chile for providing the exchange rate API
- XGBoost team for the excellent ML framework
- FastAPI for the modern web framework

---

## Quick Reference

### Essential Commands

```bash
# Data ingestion
python pipelines/data_ingestion_pipeline.py --source api --mode incremental

# Train model (planned)
python pipelines/training_pipeline.py

# Start API (planned)
uvicorn deployment.app:app --host 0.0.0.0 --port 8000

# Run tests (planned)
pytest tests/

# Build Docker image (planned)
docker build -t exchange-rate-predictor .
```

### Important Files

| File | Purpose |
|------|---------|
| `notebooks/notebook.ipynb` | Model development |
| `src/training/train.py` | Training script (to build) |
| `src/inference/predictor.py` | Prediction logic (to build) |
| `deployment/app.py` | API service (to build) |
| `config/model_config.yaml` | Model configuration (to build) |
| `models/model_balanced_v1.pkl` | Trained model (to save) |

### Useful Links

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [MLOps Best Practices](https://ml-ops.org/)

---

**Last Updated:** 2025-12-21
**Version:** 1.0.0
**Status:** Phase 1 Complete, Phase 2-6 Planned

**If this helped you, consider giving it a star!**