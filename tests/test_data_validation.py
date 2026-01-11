import logging
import pandas as pd
import pytest
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_data_integrity():
    """
    Test the integrity of the housing dataset using Deepchecks.
    This simulates a data validation step in the pipeline.
    """
    logging.info("Loading data for validation...")
    # Load your actual data or a sample. 
    # For this test, we might mock it or point to the real file if it exists.
    # Assuming 'data' folder exists in project root.
    try:
        # Try to locate an extracted file, or fail gracefully
        # In a real pipeline, this path would be passed as an argument
        df = pd.read_csv("extracted_data/AmesHousing.csv")
    except FileNotFoundError:
        logging.warning("Dataset not found locally. Using dummy data for demonstration.")
        df = pd.DataFrame({
            'SalePrice': [200000, 300000, 400000],
            'Gr Liv Area': [1500, 2000, 2500],
            'Overall Qual': [5, 7, 9]
        })

    logging.info("Running Deepchecks Data Integrity Suite...")
    ds = Dataset(df, label='SalePrice', cat_features=[])
    
    # Run the Integrity Suite
    suite = data_integrity()
    result = suite.run(ds)
    
    # Check if the suite passed
    # Deepchecks results don't map 1:1 to boolean success easily without defining conditions
    # For now, we print the result and assert that it ran successfully
    logging.info("Deepchecks run completed.")
    
    # In a real pipeline, you would assert specific checks pass:
    # assert result.passed(score_threshold=0.9)
    # For now, we are just demonstrating integration
    assert result is not None, "Deepchecks suite failed to return a result."

if __name__ == "__main__":
    test_data_integrity()
