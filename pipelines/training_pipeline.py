from steps.data_ingestion_step import data_ingestion_step
# from steps.data_drift_step import data_drift_step  # Note: Requires 'evidently' installed in environment
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step


@pipeline(
    # define the model associated with this pipeline
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)

def ml_pipeline(
    enable_cache: bool = False,
    enable_tuning: bool = False,
):
    """
    Define an end-to-end machine learning pipeline.

    The pipeline includes the following steps:
    1. Data Ingestion
    2. Handling Missing Values
    3. Feature Engineering
    4. Outlier Detection
    5. Data Splitting
    6. Model Building (with optional Tuning)
    7. Model Evaluation

    And you need to implement each step in the 'steps' directory.
    Each step should be defined as a ZenML step using the @step decorator.
    
    """

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="./data/archive.zip"
    )

    # --------------------------------------------------------------------------------
    # NEW: Data Drift Check Integration (Simulated)
    # --------------------------------------------------------------------------------
    # from steps.data_drift_step import data_drift_step
    # Note: ZenML steps must be imported at top level, but for this demo update
    # we assume we would import it. We are simulating the "call".
    #
    # drift_result = data_drift_step(
    #     reference_dataset=raw_data,  # In reality, fetch an old artifact
    #     current_dataset=raw_data     # Current incoming data
    # )
    # --------------------------------------------------------------------------------

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train, enable_tuning=enable_tuning)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
