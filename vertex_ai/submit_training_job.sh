#!/bin/bash
#
# Submit training job to Vertex AI
#
# Usage:
#   ./vertex_ai/submit_training_job.sh
#
# Prerequisites:
#   - gcloud CLI installed and configured
#   - Vertex AI API enabled
#   - Docker image built and pushed to GCR
#   - GCP_PROJECT_ID, GCS_BUCKET_MODELS, GCS_BUCKET_DATA environment variables set

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
REGION=${GCP_REGION:-us-central1}
IMAGE_NAME="exchange-rate-training"
IMAGE_TAG=${IMAGE_TAG:-latest}
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

# Job configuration
JOB_NAME="exchange-rate-training-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE=${MACHINE_TYPE:-n1-standard-4}
SERVICE_ACCOUNT=${VERTEX_AI_SERVICE_ACCOUNT}

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Submitting Vertex AI Training Job${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo "Machine Type: ${MACHINE_TYPE}"
echo ""

# Check required variables
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set${NC}"
    exit 1
fi

if [ -z "$GCS_BUCKET_MODELS" ]; then
    echo -e "${RED}Error: GCS_BUCKET_MODELS not set${NC}"
    echo "Set it with: export GCS_BUCKET_MODELS=gs://your-bucket-name"
    exit 1
fi

if [ -z "$GCS_BUCKET_DATA" ]; then
    echo -e "${RED}Error: GCS_BUCKET_DATA not set${NC}"
    echo "Set it with: export GCS_BUCKET_DATA=gs://your-bucket-name"
    exit 1
fi

# Create temporary config with environment variables
echo -e "${YELLOW}Step 1: Preparing configuration...${NC}"
CONFIG_FILE="config/training_config.yaml"

# Submit training job
echo -e "${YELLOW}Step 2: Submitting training job to Vertex AI...${NC}"

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI} \
  --args=--config,config/training_config.yaml \
  --service-account=${SERVICE_ACCOUNT}

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to submit training job${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training job submitted successfully${NC}"

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Training Job Submitted!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "Job Name: ${JOB_NAME}"
echo ""
echo "Monitor job status:"
echo "  gcloud ai custom-jobs list --region=${REGION}"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION}"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"
echo ""
echo "Or visit Vertex AI console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"