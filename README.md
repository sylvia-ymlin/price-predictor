# Housing Prices Predictor System

## Abstract

This project presents an end-to-end Machine Learning Operations (MLOps) system designed to predict housing prices using the Ames Housing dataset. The primary objective is to demonstrate a production-ready machine learning pipeline that incorporates industry-standard practices, including modular software design, experiment tracking, model deployment, and continuous integration. The system leverages ZenML for pipeline orchestration and MLflow for experiment tracking and model serving, ensuring reproducibility and scalability.

**Live Demo:** https://prices-predictor-api-52bpgfwy6q-uc.a.run.app/docs

## Methodology

The developed system follows a modular architecture where each stage of the machine learning lifecycle is encapsulated as a distinct step. This design promotes maintainability and allows for the flexible interchange of components.

### Architecture

The pipeline consists of four main stages: Data Ingestion, Preprocessing, Model Training, and Deployment.

![Architecture Diagram](docs/architecture_diagram.png)

### Design Patterns

The **Strategy Pattern** is employed throughout the codebase to define families of algorithms, encapsulate each one, and make them interchangeable. This is particularly evident in the model building phase, where different regression algorithms (Linear Regression, XGBoost, Random Forest) can be selected without altering the pipeline structure.

### Technologies

- **Orchestration**: ZenML
- **Experiment Tracking**: MLflow
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, NumPy

## Experiment Setup

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/prices-predictor-system.git
   cd prices-predictor-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the ZenML stack with MLflow integration:
   ```bash
   zenml init
   zenml integration install mlflow -y
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   zenml model-deployer register mlflow_deployer --flavor=mlflow
   zenml stack register mlflow_stack \
       -a default \
       -o default \
       -e mlflow_tracker \
       -d mlflow_deployer \
       --set
   ```

### Usage

To execute the training pipeline:
```bash
python scripts/run_pipeline.py
```

To deploy the model and start the inference server:
```bash
python scripts/run_deployment.py
```

To view the MLflow experiment dashboard:
```bash
mlflow ui --backend-store-uri 'file:./mlruns'
```

To make a sample prediction against the deployed model:
```bash
python scripts/sample_predict.py
```

### Cloud Deployment

To deploy to GCP Cloud Run:
```bash
./scripts/deploy_cloudrun.sh YOUR_PROJECT_ID us-central1
```

See [docs/cloud_deployment.md](docs/cloud_deployment.md) for detailed instructions.

## Results

The models were evaluated based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the R-squared (R²) score. The performance metrics for the tested models are summarized below.

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| Linear Regression | 0.0234 | 0.153 | 0.112 | 0.847 |
| XGBoost | 0.0189 | 0.137 | 0.098 | 0.876 |
| Random Forest | 0.0195 | 0.140 | 0.101 | 0.872 |

*Note: Metrics are computed on log-transformed target variables.*

## Hyperparameter Tuning

The pipeline includes an automated hyperparameter tuning step using `GridSearchCV`. To enable it, pass `enable_tuning=True` to the pipeline:

```python
from pipelines.training_pipeline import ml_pipeline

# Run calibration with hyperparameter search (takes longer)
ml_pipeline(enable_tuning=True)
```

This will automatically search for the best parameters (e.g., `n_estimators`, `learning_rate`) for the selected model type and log the results to MLflow.

## A/B Testing Strategy

The project supports automated Green/Blue deployments with traffic splitting via Cloud Run.

```bash
# Deploy two versions (v1 and v2) and split traffic 50/50
./scripts/deploy_ab_test.sh <GCP_PROJECT_ID> <REGION>
```

This script:
1. Builds and deploys a "Green" revision (version v1)
2. Builds and deploys a "Blue" revision (version v2)
3. Configures Cloud Run to route 50% of traffic to each version
4. Allows for live performance comparison via structured logs and metrics

## Conclusion

This project successfully establishes a robust MLOps framework for housing price prediction. The integration of ZenML and MLflow provides a transparent and reproducible workflow, while the use of design patterns ensures the codebase remains adaptable to future requirements. Future work includes the integration of more advanced feature engineering techniques and the exploration of deep learning models.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
