#!/bin/bash

# Exchange Rate Pipeline - GitHub Actions Setup Script
# This script helps you set up the pipeline for automated deployment

set -e

echo "=========================================="
echo "Exchange Rate Pipeline Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
print_step "Checking required tools..."

if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git."
    exit 1
fi

print_info "All required tools are installed."
echo ""

# Get GCP Project ID
print_step "Configuring GCP Project"
read -p "Enter your GCP Project ID: " GCP_PROJECT_ID

if [ -z "$GCP_PROJECT_ID" ]; then
    print_error "Project ID cannot be empty"
    exit 1
fi

# Set the project
gcloud config set project "$GCP_PROJECT_ID"

# Enable required APIs
print_step "Enabling BigQuery API..."
gcloud services enable bigquery.googleapis.com

# Create service account
print_step "Creating service account..."
SERVICE_ACCOUNT_NAME="exchange-rate-pipeline"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
    print_info "Service account already exists."
else
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="Exchange Rate Pipeline" \
        --description="Service account for automated exchange rate data ingestion"
    print_info "Service account created."
fi

# Grant BigQuery permissions
print_step "Granting BigQuery permissions..."
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/bigquery.admin" \
    --condition=None

# Create and download key
print_step "Creating service account key..."
KEY_FILE="gcp-key.json"

if [ -f "$KEY_FILE" ]; then
    read -p "Key file already exists. Overwrite? (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        print_info "Using existing key file."
    else
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SERVICE_ACCOUNT_EMAIL"
        print_info "New key created: $KEY_FILE"
    fi
else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL"
    print_info "Key created: $KEY_FILE"
fi

# Update config.yaml
print_step "Updating config.yaml..."
sed -i.bak "s/your-gcp-project-id/${GCP_PROJECT_ID}/g" config/config.yaml
print_info "config.yaml updated"

# Get CMF API Key
echo ""
print_step "CMF Chile API Configuration"
print_info "Get your API key from: https://api.cmfchile.cl/"
read -p "Enter your CMF Chile API key: " CMF_API_KEY

if [ -z "$CMF_API_KEY" ]; then
    print_error "API key cannot be empty"
    exit 1
fi

# Display GitHub Secrets to configure
echo ""
print_step "GitHub Secrets Configuration"
echo ""
echo "=========================================="
echo "Add these secrets to your GitHub repository:"
echo "Go to: Repository → Settings → Secrets and variables → Actions"
echo "=========================================="
echo ""

echo "✅ Secret 1: GCP_PROJECT_ID"
echo "   Value: ${GCP_PROJECT_ID}"
echo ""

echo "✅ Secret 2: GCP_CREDENTIALS"
echo "   Value: (entire JSON content of ${KEY_FILE})"
echo "   Command to view: cat ${KEY_FILE}"
echo ""

echo "✅ Secret 3: CMF_API_KEY"
echo "   Value: ${CMF_API_KEY}"
echo ""

echo "=========================================="
echo "⚠️  IMPORTANT: Only these 3 secrets are needed!"
echo "    The workflow ONLY uses Exchange Rate API"
echo "=========================================="
echo ""

# Create a temporary file with instructions
SECRETS_FILE="github_secrets_instructions.txt"
cat > "$SECRETS_FILE" << EOF
========================================
GitHub Secrets Configuration
========================================

🔐 Add these 3 SECRETS to your GitHub repository:

Repository → Settings → Secrets and variables → Actions → New repository secret

---

✅ Secret 1: GCP_PROJECT_ID

   Value: ${GCP_PROJECT_ID}

---

✅ Secret 2: GCP_CREDENTIALS

   Value: (Copy the ENTIRE JSON content below)

---START JSON---
$(cat "$KEY_FILE")
---END JSON---

---

✅ Secret 3: CMF_API_KEY

   Value: ${CMF_API_KEY}

---

========================================
⚠️  IMPORTANT:
========================================

• Only these 3 secrets are required
• The workflow is configured for Exchange Rate API only
• GCP_CREDENTIALS must include the ENTIRE JSON

========================================
Next Steps:
========================================

1. Add the 3 secrets to GitHub (see above)
2. Push to GitHub: git add . && git commit -m "Setup" && git push
3. Go to GitHub → Actions → "Exchange Rate Data Ingestion"
4. Click "Run workflow" → Select mode: incremental
5. Check results in BigQuery

For detailed instructions, see: GITHUB_SETUP.md
========================================
EOF

print_info "Instructions saved to: ${SECRETS_FILE}"

# Test local pipeline
echo ""
read -p "Do you want to test the pipeline locally? (y/n): " test_local

if [ "$test_local" = "y" ]; then
    print_step "Testing pipeline locally..."

    # Create secrets.yaml for local testing
    cat > config/secrets.yaml << EOF
sql:
  password_prod: ""
  password_dev: ""

api:
  api_key: "${CMF_API_KEY}"

gcp:
  credentials_path: "${KEY_FILE}"
EOF

    # Set up credentials
    export GOOGLE_APPLICATION_CREDENTIALS="${KEY_FILE}"

    # Install dependencies
    print_step "Installing dependencies..."
    pip install -r requirements.txt

    # Run pipeline
    print_step "Running pipeline in incremental mode..."
    python pipelines/exchange_rate_pipeline.py --mode incremental

    print_info "Local test completed! Check the output above for any errors."
fi

echo ""
print_step "Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Review ${SECRETS_FILE} for GitHub secrets"
echo "2. Add secrets to GitHub: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions"
echo "3. Push code to GitHub: git add . && git commit -m 'Configure pipeline' && git push"
echo "4. Manually trigger workflow in GitHub Actions"
echo ""
print_info "For detailed instructions, see DEPLOYMENT.md"
