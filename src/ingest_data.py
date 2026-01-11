import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    """
    Abstract base class for data ingestion strategies.

    This class defines the interface for ingesting data from various sources.
    Subclasses must implement the `ingest` method to handle specific data formats.
    """

    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given file.

        Args:
            file_path (str): The file path to the data source.

        Returns:
            pd.DataFrame: The ingested data as a pandas DataFrame.
        """
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    """
    Concrete strategy for ingesting data from ZIP archives.

    This class extracts a ZIP file and loads the contained CSV file into a
    pandas DataFrame. It assumes the ZIP archive contains exactly one CSV file.
    """

    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Extracts a .zip file and returns the content as a pandas DataFrame.

        Args:
            file_path (str): The path to the .zip file.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            ValueError: If the file is not a .zip file or if multiple/no CSVs are found.
            FileNotFoundError: If no CSV file is found in the extracted data.
        """
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    """
    Factory class for creating DataIngestor instances.
    """

    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on file extension.

        Args:
            file_extension (str): The file extension (e.g., '.zip').

        Returns:
            DataIngestor: An instance of a concrete DataIngestor subclass.

        Raises:
            ValueError: If no ingestor is available for the given extension.
        """
        # here we only implement ZipDataIngestor, but more can be added to support other formats
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


# Example usage:
if __name__ == "__main__":
    # Specify the file path
    file_path = "../data/archive.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    # Now df contains the DataFrame from the extracted CSV
    print(df.head())  # Display the first few rows of the DataFrame
