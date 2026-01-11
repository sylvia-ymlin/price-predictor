#!/bin/bash
#
# GCP Cloud Run Deployment Script
# Deploys the Housing Prices Predictor API to Google Cloud Run
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Docker installed locally
#   3. GCP project with billing enabled
#
# Usage:
#   ./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION]
#
# Example:
#   ./scripts/deploy_cloudrun.sh my-gcp-project us-central1

set -e

# Configuration
PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
SERVICE_NAME="prices-predictor-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=========================================="
echo "GCP Cloud Run Deployment"
echo "=========================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service Name: ${SERVICE_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "=========================================="

# Validate project
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project specified."
    echo "Usage: ./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION]"
    exit 1
fi

# Step 1: Enable required APIs
echo ""
echo "[1/5] Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com --project=${PROJECT_ID}
gcloud services enable run.googleapis.com --project=${PROJECT_ID}
gcloud services enable containerregistry.googleapis.com --project=${PROJECT_ID}

# Step 2: Build and push Docker image using Cloud Build
echo ""
echo "[2/5] Building Docker image with Cloud Build..."
gcloud builds submit --tag ${IMAGE_NAME} --project=${PROJECT_ID}

# Step 3: Deploy to Cloud Run
echo ""
echo "[3/5] Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 1Gi \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 2 \
    --project=${PROJECT_ID}

# Step 4: Get service URL
echo ""
echo "[4/5] Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

# Step 5: Test the deployment
echo ""
echo "[5/5] Testing deployment..."
echo "Health check: ${SERVICE_URL}/health"
curl -s "${SERVICE_URL}/health" | head -c 200
echo ""

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test prediction endpoint:"
echo "  curl -X POST ${SERVICE_URL}/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"features\": {\"Gr Liv Area\": 1500, \"Overall Qual\": 7}}'"
echo ""
echo "API Documentation: ${SERVICE_URL}/docs"
echo ""
