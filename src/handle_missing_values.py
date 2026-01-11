import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    """
    Abstract base class for missing value handling strategies.

    This class defines the interface for handling missing values in dataframes.
    Subclasses must implement the `handle` method.
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy for dropping rows or columns with missing values.
    """

    def __init__(self, axis: int = 0, thresh: int = None):
        """
        Initialize the DropMissingValuesStrategy.

        Args:
            axis (int): 0 to drop rows, 1 to drop columns. Defaults to 0.
            thresh (int, optional): Require that many non-NA values. Defaults to None.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


# Concrete Strategy for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy for filling missing values with a specified method or value.
    """

    def __init__(self, method: str = "mean", fill_value: any = None):
        """
        Initialize the FillMissingValuesStrategy.

        Args:
            method (str): The method to fill ('mean', 'median', 'mode', 'constant'). Defaults to 'mean'.
            fill_value (any, optional): The value to use when method is 'constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned


# Context Class for Handling Missing Values
class MissingValueHandler:
    """
    Context class for executing missing value handling strategies.
    """

    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initialize the MissingValueHandler with a specific strategy.

        Args:
            strategy (MissingValueHandlingStrategy): The strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Args:
            strategy (MissingValueHandlingStrategy): The new strategy to use.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling strategy.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)

    pass
