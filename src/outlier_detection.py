import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    """
    Abstract base class for outlier detection strategies.

    This class defines the interface for detecting outliers in datasets.
    Subclasses must implement the `detect_outliers` method.
    """

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the given DataFrame.

        Args:
            df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        pass


# Concrete Strategy for Z-Score Based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """
    Strategy for Z-Score based outlier detection.

    This strategy detects outliers based on the number of standard deviations
    a data point is from the mean.
    """

    def __init__(self, threshold: float = 3):
        """
        Initialize the ZScoreOutlierDetection strategy.

        Args:
            threshold (float): The Z-score threshold for defining outliers. Defaults to 3.
        """
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers using the Z-score method.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        logging.info("Detecting outliers using the Z-score method.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outliers


# Concrete Strategy for IQR Based Outlier Detection
class IQROutlierDetection(OutlierDetectionStrategy):
    """
    Strategy for IQR (Interquartile Range) based outlier detection.

    This strategy detects outliers that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    """

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers using the IQR method.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        logging.info("Detecting outliers using the IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outliers detected using the IQR method.")
        return outliers


# Context Class for Outlier Detection and Handling
class OutlierDetector:
    """
    Context class for outlier detection and handling.
    """

    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Initialize the OutlierDetector with a specific strategy.

        Args:
            strategy (OutlierDetectionStrategy): The strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        """
        Sets a new strategy for the OutlierDetector.

        Args:
            strategy (OutlierDetectionStrategy): The new strategy to use.
        """
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the outlier detection strategy.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: A boolean dataframe indicating outliers.
        """
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove", **kwargs) -> pd.DataFrame:
        """
        Handles outliers by removing or capping them.

        Args:
            df (pd.DataFrame): The input dataframe.
            method (str): The method to handle outliers ('remove' or 'cap'). Defaults to 'remove'.
            **kwargs: Additional arguments (not currently used).

        Returns:
            pd.DataFrame: The dataframe with outliers handled.
        """
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # df = pd.read_csv("../extracted_data/AmesHousing.csv")
    # df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    # outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # # Detect and handle outliers
    # outliers = outlier_detector.detect_outliers(df_numeric)
    # df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

    # print(df_cleaned.shape)
    # # Visualize outliers in specific features
    # # outlier_detector.visualize_outliers(df_cleaned, features=["SalePrice", "Gr Liv Area"])
    pass


#    SalePrice  Gr Liv Area
# 0     200000        1500
# 1     300000        2000
# 2     400000        2500
# 3     500000        3000
# 4    1000000        4000
# 5    2000000       10000


#    SalePrice  Gr Liv Area
# 0     200000        1500
# 1     300000        2000
# 2     400000        2500
# 3     500000        3000
# 4    1000000        4000
# 5    1000000        4000
