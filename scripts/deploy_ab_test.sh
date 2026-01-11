#!/bin/bash
set -e

# Configuration
PROJECT_ID=$1
REGION=$2
SERVICE_NAME="prices-predictor-api"
REPO_NAME="prices-predictor-api"

if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ]; then
    echo "Usage: ./deploy_ab_test.sh <PROJECT_ID> <REGION>"
    exit 1
fi

echo "=========================================="
echo "GCP Cloud Run A/B Testing Deployment"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"
echo "=========================================="

# Create temporary cloudbuild configuration
cat > cloudbuild_temp.yaml <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [ 'build', '-t', '\$_IMAGE_NAME', '--build-arg', 'MODEL_VERSION=\$_MODEL_VERSION', '.' ]
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', '\$_IMAGE_NAME']
EOF

# --- Phase 1: Deploy Baseline (Green / v1) ---
echo ""
echo "[1/4] Deploying Baseline (v1)..."
IMAGE_V1="gcr.io/$PROJECT_ID/$REPO_NAME:v1"

echo "Building v1 image..."
gcloud builds submit --config cloudbuild_temp.yaml \
    --substitutions _MODEL_VERSION="v1",_IMAGE_NAME="$IMAGE_V1" \
    --project "$PROJECT_ID"

echo "Deploying v1 revision..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_V1" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --platform managed \
    --allow-unauthenticated \
    --tag "green"

# --- Phase 2: Deploy Challenger (Blue / v2) ---
echo ""
echo "[2/4] Deploying Challenger (v2)..."
IMAGE_V2="gcr.io/$PROJECT_ID/$REPO_NAME:v2"

echo "Building v2 image..."
gcloud builds submit --config cloudbuild_temp.yaml \
    --substitutions _MODEL_VERSION="v2",_IMAGE_NAME="$IMAGE_V2" \
    --project "$PROJECT_ID"

echo "Deploying v2 revision (no-traffic initially)..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_V2" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --platform managed \
    --allow-unauthenticated \
    --tag "blue" \
    --no-traffic

# --- Phase 3: Split Traffic ---
echo ""
echo "[3/4] Splitting Traffic 50/50..."
gcloud run services update-traffic "$SERVICE_NAME" \
    --to-tags green=50,blue=50 \
    --region "$REGION" \
    --project "$PROJECT_ID"

# --- Phase 4: Clean up ---
rm cloudbuild_temp.yaml

echo ""
echo "=========================================="
echo "A/B Test Deployment Complete!"
echo "=========================================="
echo "Service is now serving 50% traffic to Green (v1) and 50% to Blue (v2)."
echo "You can check the 'model_version' in the JSON logs or API response."
echo "URL: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')"
