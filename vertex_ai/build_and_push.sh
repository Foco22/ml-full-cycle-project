#!/bin/bash
#
# Build Docker image and push to Google Container Registry (GCR)
#
# Usage:
#   ./vertex_ai/build_and_push.sh
#
# Prerequisites:
#   - gcloud CLI installed and configured
#   - Docker installed
#   - GCP_PROJECT_ID environment variable set

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
IMAGE_NAME="exchange-rate-training"
IMAGE_TAG=${IMAGE_TAG:-latest}
REGION=${GCP_REGION:-us-central1}

# Full image path
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Building and Pushing Vertex AI Training Image${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "Project ID: ${PROJECT_ID}"
echo "Image Name: ${IMAGE_NAME}"
echo "Image Tag: ${IMAGE_TAG}"
echo "Image URI: ${IMAGE_URI}"
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set${NC}"
    echo "Set it with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Authenticate Docker with GCR
echo -e "${YELLOW}Step 1: Authenticating Docker with GCR...${NC}"
gcloud auth configure-docker gcr.io --quiet

# Build Docker image
echo -e "${YELLOW}Step 2: Building Docker image...${NC}"
docker build \
    -f vertex_ai/Dockerfile \
    -t ${IMAGE_URI} \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Push to GCR
echo -e "${YELLOW}Step 3: Pushing image to GCR...${NC}"
docker push ${IMAGE_URI}

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker push failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Image pushed successfully${NC}"

# Test image locally (optional)
echo ""
read -p "Test image locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Testing image locally...${NC}"
    docker run --rm \
        -v $(pwd)/config:/app/config \
        -e GCP_PROJECT_ID=${PROJECT_ID} \
        ${IMAGE_URI} \
        --config config/training_config.yaml \
        --local
fi

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Build and Push Complete!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "Image URI: ${IMAGE_URI}"
echo ""
echo "Next steps:"
echo "  1. Update config/training_config.yaml with this image URI"
echo "  2. Run: ./vertex_ai/submit_training_job.sh"