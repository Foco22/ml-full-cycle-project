# Vertex AI Training Setup

This folder contains scripts and configurations for training ML models on Google Cloud Vertex AI.

## Prerequisites

### 1. Enable Required GCP APIs

```bash
# Enable all required APIs at once
gcloud services enable \
  artifactregistry.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  --project=ml-project-479423
```

Or enable them individually:
```bash
gcloud services enable artifactregistry.googleapis.com --project=ml-project-479423
gcloud services enable containerregistry.googleapis.com --project=ml-project-479423
gcloud services enable cloudbuild.googleapis.com --project=ml-project-479423
gcloud services enable aiplatform.googleapis.com --project=ml-project-479423
```

### 2. Create GCS Bucket

```bash
# Create the bucket (if not exists)
gsutil mb -p ml-project-479423 -l us-central1 gs://ml-project-forecast

# Create folder structure
gsutil mkdir gs://ml-project-forecast/models
gsutil mkdir gs://ml-project-forecast/data
gsutil mkdir gs://ml-project-forecast/artifacts
gsutil mkdir gs://ml-project-forecast/logs
```

### 3. Service Account Permissions

Make sure your service account `ml-project-cs@ml-project-479423.iam.gserviceaccount.com` has these roles:

```bash
# Grant required roles
gcloud projects add-iam-policy-binding ml-project-479423 \
  --member="serviceAccount:ml-project-cs@ml-project-479423.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding ml-project-479423 \
  --member="serviceAccount:ml-project-cs@ml-project-479423.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding ml-project-479423 \
  --member="serviceAccount:ml-project-cs@ml-project-479423.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"
```

### 4. GitHub Secrets

Configure these secrets in your GitHub repository:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `GCP_CREDENTIALS` | Service account JSON | Full JSON key file |
| `GCP_PROJECT_ID` | `ml-project-479423` | Project ID |
| `VERTEX_AI_SERVICE_ACCOUNT` | `ml-project-cs@ml-project-479423.iam.gserviceaccount.com` | Service account email |
| `GCS_BUCKET_DATA` | `gs://ml-project-forecast/data` | Data bucket path |
| `GCS_BUCKET_MODELS` | `gs://ml-project-forecast/models` | Models bucket path |
| `GCS_BUCKET_ARTIFACTS` | `gs://ml-project-forecast/artifacts` | Artifacts bucket path |
| `GCS_BUCKET_LOGS` | `gs://ml-project-forecast/logs` | Logs bucket path |

## Usage

### Option 1: GitHub Actions (Automated)

The workflow automatically triggers on push to paths:
- `src/training/**`
- `config/**`
- `vertex_ai/**`

### Option 2: Manual GitHub Actions Trigger

1. Go to Actions tab in GitHub
2. Select "Vertex AI Model Training" workflow
3. Click "Run workflow"
4. Select branch and options
5. Click "Run workflow"

### Option 3: Local Execution

```bash
# 1. Build and push Docker image
./vertex_ai/build_and_push.sh

# 2. Submit training job
python vertex_ai/submit_training_job.py \
  --project-id ml-project-479423 \
  --region us-central1 \
  --image-uri gcr.io/ml-project-479423/exchange-rate-training:latest \
  --machine-type n1-standard-4 \
  --service-account ml-project-cs@ml-project-479423.iam.gserviceaccount.com
```

## Monitoring

### View Training Jobs
```bash
# List all training jobs
gcloud ai custom-jobs list --region=us-central1 --project=ml-project-479423

# View specific job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### Vertex AI Console
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ml-project-479423

### View Model Artifacts
```bash
# List trained models
gsutil ls gs://ml-project-forecast/models/model_balanced/

# Download specific model
gsutil cp -r gs://ml-project-forecast/models/model_balanced/YYYYMMDD_HHMMSS/ ./local_model/
```

## Troubleshooting

### Error: API not enabled
```bash
# Enable the missing API
gcloud services enable API_NAME --project=ml-project-479423
```

### Error: Permission denied
Check service account has required roles (see Prerequisites #3)

### Error: Bucket not found
```bash
# Create the bucket
gsutil mb -p ml-project-479423 -l us-central1 gs://ml-project-forecast
```

## Files

- `Dockerfile` - Container definition for training
- `build_and_push.sh` - Script to build and push Docker image
- `submit_training_job.py` - Python script to submit Vertex AI job
- `submit_training_job.sh` - Bash script to submit Vertex AI job
- `requirements.txt` - Python dependencies for training container

## Training Output Structure

```
gs://ml-project-forecast/
├── models/model_balanced/YYYYMMDD_HHMMSS/
│   ├── model.pkl.gz       # Trained model
│   ├── metadata.json      # Model metadata and metrics
│   ├── features.json      # Feature list
│   └── config.yaml        # Training configuration
├── data/training/YYYYMMDD_HHMMSS/
│   ├── train_data.csv     # Training dataset
│   └── test_data.csv      # Test dataset
├── artifacts/model_balanced/YYYYMMDD_HHMMSS/
│   ├── metrics.json       # Detailed metrics
│   └── plots/             # Validation plots
└── logs/training/YYYYMMDD_HHMMSS/
    └── training.log       # Training logs
```