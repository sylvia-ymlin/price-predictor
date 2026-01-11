# Cloud Deployment Guide

This guide documents the deployment of the Housing Prices Predictor API to Google Cloud Run.

---

## Architecture

```
[Local Development] --> [Cloud Build] --> [Container Registry] --> [Cloud Run]
                                                                        |
                                                                   [HTTPS URL]
                                                                        |
                                                                [Public Access]
```

---

## Prerequisites

1. GCP account with billing enabled
2. gcloud CLI installed and authenticated
3. Docker (for local testing)

### Install gcloud CLI

```bash
# macOS
brew install --cask google-cloud-sdk

# Verify installation
gcloud --version
```

### Authenticate

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

---

## Deployment

### Option 1: Automated Script

```bash
chmod +x scripts/deploy_cloudrun.sh
./scripts/deploy_cloudrun.sh YOUR_PROJECT_ID us-central1
```

### Option 2: Manual Steps

```bash
# 1. Set project
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="prices-predictor-api"

# 2. Enable APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 3. Build image with Cloud Build
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# 4. Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated

# 5. Get URL
gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format="value(status.url)"
```

---

## Deployed Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/docs` | GET | Swagger UI |

### Example Request

```bash
SERVICE_URL="https://prices-predictor-api-xxxxx.a.run.app"

curl -X POST "${SERVICE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "features": {
            "Gr Liv Area": 1500,
            "Overall Qual": 7,
            "Total Bsmt SF": 1000,
            "Garage Cars": 2
        }
    }'
```

### Expected Response

```json
{
    "predicted_price": 185000.50,
    "model_version": "1.0.0",
    "timestamp": "2026-01-07T21:15:00Z"
}
```

---

## Configuration

### Cloud Run Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Memory | 1Gi | ML model requires memory |
| Timeout | 300s | Allow time for cold start |
| Concurrency | 80 | Requests per instance |
| Min instances | 0 | Scale to zero when idle |
| Max instances | 2 | Limit costs |

### Environment Variables

Set via Cloud Run console or CLI:

```bash
gcloud run services update ${SERVICE_NAME} \
    --set-env-vars="MLFLOW_TRACKING_URI=..." \
    --region ${REGION}
```

---

## Monitoring

### View Logs

```bash
gcloud run services logs read ${SERVICE_NAME} --region ${REGION}
```

### View Metrics

Access via GCP Console: Cloud Run > prices-predictor-api > Metrics

Key metrics:
- Request count
- Latency (p50, p95, p99)
- Instance count
- Memory utilization

---

## Cost Estimation

Cloud Run pricing (as of 2024):

| Resource | Free Tier | After Free Tier |
|----------|-----------|-----------------|
| Requests | 2 million/month | $0.40/million |
| CPU | 180,000 vCPU-seconds/month | $0.00002400/vCPU-second |
| Memory | 360,000 GiB-seconds/month | $0.00000250/GiB-second |

For a demo project with minimal traffic, expected cost: $0-5/month

---

## Troubleshooting

### Cold Start Latency

First request after idle may take 5-10 seconds due to container startup.

Solution: Set `--min-instances 1` for always-warm service (increases cost).

### Build Failures

Check Cloud Build logs:
```bash
gcloud builds list --limit=5
gcloud builds log BUILD_ID
```

### Permission Errors

Ensure Cloud Build service account has required permissions:
```bash
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin"
```

---

## Cleanup

To delete the deployment and avoid charges:

```bash
# Delete Cloud Run service
gcloud run services delete prices-predictor-api --region us-central1

# Delete container images
gcloud container images delete gcr.io/${PROJECT_ID}/prices-predictor-api --force-delete-tags
```
