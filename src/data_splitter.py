import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Splitting Strategy
class DataSplittingStrategy(ABC):
    """
    Abstract base class for data splitting strategies.

    This class defines the interface for splitting data into training and testing sets.
    Subclasses must implement the `split_data` method.
    """

    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Abstract method to split the data into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            target_column (str): The name of the target column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                A tuple containing (X_train, X_test, y_train, y_test).
        """
        pass


# Concrete Strategy for Simple Train-Test Split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Strategy for simple train-test splitting.

    This strategy randomly partitions the data into training and testing sets
    based on a specified test size.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the SimpleTrainTestSplitStrategy.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the data into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            target_column (str): The name of the target column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                A tuple containing (X_train, X_test, y_train, y_test).
        """
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


# Context Class for Data Splitting
class DataSplitter:
    """
    Context class for data splitting operations.

    This class delegates the data splitting logic to the configured
    DataSplittingStrategy.
    """

    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific strategy.

        Args:
            strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Args:
            strategy (DataSplittingStrategy): The new data splitting strategy.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Executes the data splitting using the current strategy.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            target_column (str): The name of the target column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                A tuple containing (X_train, X_test, y_train, y_test).
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)


# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')

    # Initialize data splitter with a specific strategy
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    pass
