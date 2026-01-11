from zenml import step
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

@step
def data_drift_step(
    reference_dataset: pd.DataFrame,
    current_dataset: pd.DataFrame,
) -> dict:
    """
    Detects data drift between a reference dataset and the current dataset.
    
    Args:
        reference_dataset: The dataset used for training (or a baseline).
        current_dataset: The new batch of data to check for drift.
        
    Returns:
        dict: A dictionary containing drift metrics (e.g., drift_share).
    """
    # Initialize the specific Evidently Report
    report = Report(metrics=[
        DataDriftPreset(), 
    ])
    
    # Run the drift calculation
    report.run(reference_data=reference_dataset, current_data=current_dataset)
    
    # Extract the result as a python dictionary
    result = report.as_dict()
    
    # Log key metrics
    drift_share = result['metrics'][0]['result']['drift_share']
    number_of_drifted_columns = result['metrics'][0]['result']['number_of_drifted_columns']
    
    print(f"Data Drift Report Completed.")
    print(f"Drift Share: {drift_share}")
    print(f"Drifted Columns: {number_of_drifted_columns}")
    
    return result
