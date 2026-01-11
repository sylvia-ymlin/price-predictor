# Interview Preparation Guide: Housing Prices Predictor System

This document provides structured preparation for technical interviews based on this MLOps project.

---

## 0. MLE Capability Assessment

Self-assessment of this project against typical ML Engineer job requirements:

| Capability | Current Status | MLE Expectation | Gap Assessment |
|------------|---------------|-----------------|----------------|
| MLOps Pipeline | ZenML + MLflow | Required | Satisfied |
| Model Serving | FastAPI REST | Required | Satisfied |
| Containerization | Docker Compose | Required | Satisfied |
| CI/CD | GitHub Actions | Required | Satisfied |
| Testing | 54+ Pytest cases | Required | Satisfied |
| Model Monitoring | Evidently drift detection | Strongly Expected | Satisfied |
| Hyperparameter Tuning | GridSearchCV in Pipeline | Expected | Satisfied |
| Feature Store | None | Bonus | Consider Adding |
| Model Interpretability | SHAP integration | Expected | Satisfied |
| A/B Testing / Shadow Mode | Cloud Run Traffic Split | Bonus | Satisfied |
| Cloud Deployment | GCP Cloud Run | Strongly Expected | Satisfied |
| Real-time Inference | FastAPI REST Endpoint | Role-dependent | Satisfied |

### Optimization Roadmap

Completed:
1. Model Interpretability - SHAP for feature importance visualization (Done)
2. Cloud Deployment - GCP Cloud Run for live demo (Done)
3. Web Frontend - Modern HTML/CSS/JS interface for predictions (Done)
4. Hyperparameter Tuning - Integrated GridSearchCV in ZenML pipeline (Done)
5. Performance Monitoring - Structured Logging & Prometheus Metrics (Done)
6. A/B Testing - Automated Green/Blue deployment with Traffic Splitting (Done)

**Live Demo:** https://prices-predictor-api-52bpgfwy6q-uc.a.run.app

Priority 2 (Nice to Have):
7. Feature Store - Feast integration for feature management
7. Feature Store - Feast integration for feature management

---

## 1. Project Context

- **Timeline:** April 2024 – July 2024 (3 Months)
- **Role:** ML Engineer (End-to-End Ownership)
- **Goal:** Build a scalable, automated pricing engine to replace legacy manual estimation processes.

## 2. Elevator Pitch (30 seconds)

"I built a production-grade housing price prediction system using ZenML and MLflow, deployed live on GCP Cloud Run. The project features a fully automated MLOps pipeline with **hyperparameter tuning** and **data drift monitoring**. On the serving side, I implemented a monitored FastAPI service with **structured logging** and **Prometheus metrics** for full observability. I also optimized the architecture by utilizing Docker build-time training to embed the model, enabling a robust, serverless deployment with a React-like frontend."

---

## 2. Technical Deep Dive

### 2.1 Architecture Overview

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| Orchestration | ZenML | Pipeline DAG, step caching, artifact management |
| Tracking | MLflow | Parameter logging, metric recording, model registry |
| Serving | FastAPI | REST API endpoint for inference |
| Observability | Prometheus + JSON Logs | Metrics scraping and structured logging |
| Quality | Evidently AI + Pytest | Data drift detection + unit testing |
| Infrastructure | Docker + Cloud Run | Serverless container deployment |
| Explainability | SHAP | Feature importance, prediction explanations |
| Cloud | GCP Cloud Run | Serverless container deployment |
| Frontend | HTML/CSS/JS | User-facing prediction interface |

### 2.2 Key Design Patterns

**Strategy Pattern (src/model_building.py)**

Problem: Hardcoded if-else logic for model selection creates brittle, difficult-to-maintain code.

Solution: Abstract `ModelBuildingStrategy` interface with concrete implementations for each algorithm.

```python
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train, y_train) -> RegressorMixin:
        pass

class XGBoostStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train, y_train):
        # Implementation
```

Result: Adding CatBoost or LightGBM requires only a new strategy class; zero changes to pipeline code.

**Deployment Gating (Continuous Deployment)**

Logic: Deployment pipeline compares candidate model MSE against production model MSE.

Action: Promotes to production only if `MSE_new < MSE_production`.

Purpose: Prevents performance regression in production.

**Data Drift Detection**

Tool: Evidently AI within a ZenML step.

Logic: Computes statistical distance between training data distribution and incoming inference data.

Purpose: Alerts when model effectiveness may decay due to shifting real-world data distributions.

### 2.3 Performance Metrics

| Model | MSE | RMSE | R-squared |
|-------|-----|------|-----------|
| XGBoost (Best) | 0.0189 | 0.137 | 0.876 |
| Random Forest | 0.0195 | 0.140 | 0.872 |
| Linear Regression | 0.0234 | 0.153 | 0.847 |

Drift Sensitivity: Configured to alert if more than 50% of columns show statistical drift.

---

## 3. Behavioral Questions (STAR Format)

### Q: Tell me about a technical challenge you faced.

**Situation**: During initial deployment, the API returned static dummy predictions because the model loading logic was fragile and not connected to the actual trained model.

**Task**: Connect the FastAPI application to the MLflow Model Registry to serve real predictions.

**Action**: Refactored `src/app.py` to integrate with MLflow Model Registry. Added exception handling to ensure the health endpoint only reports "Healthy" when a valid model artifact is successfully loaded.

**Result**: The system now dynamically serves the latest production-stage model without manual server restarts. Predictions are deterministic and traceable to specific training runs.

---

### Q: How did you improve or optimize the system?

**Situation**: The original pipeline was functional but blind to data quality issues. It would continue training on potentially corrupted or drifted data without any warning.

**Task**: Add observability to detect when incoming data distribution deviates from training data.

**Action**: Introduced a `data_drift_step` using Evidently AI. This step runs after data ingestion and compares current batch statistics against a reference dataset.

**Result**: The pipeline now generates a drift report for every run. If the input data distribution (e.g., "Gr Liv Area" feature) starts shifting, the system flags this before model training begins.

---

### Q: Describe a time you had to make a difficult technical decision.

**Situation**: Choosing between a heavyweight MLOps platform (Kubeflow) and a lightweight solution (ZenML + MLflow) for pipeline orchestration.

**Task**: Select an architecture that balances production readiness with development velocity for a single-developer project.

**Action**: Evaluated trade-offs:
- Kubeflow: Kubernetes-native, enterprise features, but requires dedicated infra team
- ZenML: Python-first, easy local development, can migrate to Kubeflow later

Chose ZenML + MLflow because the same code runs locally and can be switched to Kubeflow orchestrator via configuration only.

**Result**: Achieved production-grade reproducibility without the operational overhead. The system can scale to enterprise infrastructure when needed without code refactoring.

---

## 4. Code Walkthrough Scenarios

Be prepared to open and explain these files:

### Scenario 1: "Show me your training logic"

Files: `pipelines/training_pipeline.py`, `steps/model_building_step.py`

Key points to highlight:
- `@pipeline` decorator and DAG definition
- How steps pass typed artifacts (DataFrames, models) to each other
- ZenML automatic caching of unchanged steps

### Scenario 2: "How do you handle different models?"

Files: `src/model_building.py`

Key points to highlight:
- `ModelBuildingStrategy` abstract class
- Concrete strategy implementations (XGBoost, RandomForest, LinearRegression)
- `ModelBuilder` context class that switches strategies at runtime

### Scenario 3: "How is this deployed?"

Files: `docker-compose.yml`, `src/app.py`

Key points to highlight:
- Separation of `mlflow-tracking` service and `prediction-api` service
- Environment variable injection for service discovery
- Volume mounts for artifact persistence

### Scenario 4: "How do you ensure code quality?"

Files: `.github/workflows/ci.yml`, `tests/`

Key points to highlight:
- CI pipeline stages: format check, lint, type check, test
- Test coverage reporting to Codecov
- Pip caching for faster CI runs

---

## 5. Anticipated Technical Questions

### Q: Why not just use a Jupyter notebook for deployment?

Notebooks are effective for exploratory data analysis but problematic for production due to:
1. Hidden state: Cell execution order is non-deterministic
2. Version control: Git diffs on `.ipynb` files are unreadable
3. Automation: Cannot integrate into CI/CD pipelines

This MLOps system solves the 1-to-N problem of stable, reproducible deployments. Anyone can clone the repository and reproduce the exact same model artifact.

---

### Q: How do you explain model predictions to stakeholders?

Integrated SHAP (SHapley Additive exPlanations) for model interpretability. SHAP provides:

1. Global explanations: Feature importance rankings across all predictions
2. Local explanations: Why a specific house was predicted at a certain price

Example output:
```
Top 5 features for price prediction:
1. Gr Liv Area (living area square feet) - 0.42 importance
2. Overall Qual (overall quality rating) - 0.28 importance
3. Total Bsmt SF (basement area) - 0.12 importance
4. Garage Cars (garage capacity) - 0.08 importance
5. Year Built - 0.05 importance
```

For individual predictions, SHAP waterfall plots show which features pushed the price up or down from the baseline.

---

### Q: What design patterns did you use and why?

Primary pattern: Strategy Pattern for model building.

Reasoning:
1. Eliminates conditional branching in pipeline code
2. Each algorithm is encapsulated in its own class with clear responsibilities
3. Adheres to Open-Closed Principle: open for extension (new strategies), closed for modification (existing pipeline)

Secondary pattern: Template Method in preprocessing steps, where the skeleton of the algorithm is defined in the base class but specific operations are delegated to subclasses.

---

### Q: How do you prevent deploying a worse model to production?

Implemented Deployment Gating with two mechanisms:

1. Threshold gate: Model MSE must be below a configured threshold value
2. Comparison gate: Candidate model must outperform current production model

The deployment pipeline loads the current production model from MLflow Registry, evaluates both models on a holdout set, and only promotes the candidate if it demonstrates improvement.

### Q: Even if offline metrics are good, how do you ensure the model drives business value? (A/B Testing)

Offline metrics (like RMSE) don't always correlate with user satisfaction. I implemented an **Infrastructure-based A/B Testing** strategy using Cloud Run Traffic Splitting to validate business impact:

- **Scenario**: A complex XGBoost model might fit the data better (lower RMSE) but produce volatile predictions for edge cases, reducing user trust. A simpler Linear Regression model might be less "accurate" but more "predictable."
- **Execution**: I deploy both models (Green/Blue) and route 50% traffic to each.
- **Success Metric**: Instead of just accuracy, we monitor business KPIs like **"User Retention"** or **"Lead Conversion Rate"** (e.g., how many users contact an agent after seeing the price).
- **The "Why"**: This bridges the gap between Data Science metrics and Business Reality.

---

### Q: What happens if the data distribution changes after deployment?

This is the data drift problem. The system addresses it through:

1. Evidently AI integration in the training pipeline to generate baseline data profiles
2. Optional drift detection step that can compare incoming inference data against training distribution
3. Configurable thresholds to trigger alerts when drift exceeds acceptable limits

When drift is detected, the recommended action is model retraining with recent data.

---

### Q: How would you scale this system for higher throughput?

Current architecture already supports horizontal scaling:

1. Prediction API is stateless; can run multiple replicas behind a load balancer
2. MLflow tracking server is separated; inference latency is not coupled to logging
3. Docker Compose can be replaced with Kubernetes manifests for orchestrated scaling

For higher throughput specifically:
- Add async inference via task queue (Celery/Redis)
- Implement model batching for GPU efficiency
- Introduce caching layer for repeated prediction requests

---

## 6. Questions to Ask Interviewers

1. "What does your current ML deployment pipeline look like? Are there specific pain points this role would address?"

2. "How do you handle model versioning and rollback in production?"

3. "What's the team's approach to balancing experimentation speed with production stability?"

4. "Are there existing observability tools for ML systems, or is that an area for development?"
