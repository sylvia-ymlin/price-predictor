import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
class FeatureEngineeringStrategy(ABC):
    """
    Abstract base class for feature engineering strategies.

    This class defines the interface for applying feature transformations.
    Subclasses must implement the `apply_transformation` method.
    """

    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Args:
            df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
            pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
class LogTransformation(FeatureEngineeringStrategy):
    """
    Strategy for applying log transformations.

    This strategy applies a logarithmic transformation (log1p) to specified features
    to handle skewed distributions.
    """

    def __init__(self, features: list):
        """
        Initialize the LogTransformation strategy.

        Args:
            features (list): The list of features (column names) to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
class StandardScaling(FeatureEngineeringStrategy):
    """
    Strategy for applying standard scaling (z-score normalization).

    This strategy scales features to have zero mean and unit variance.
    """

    def __init__(self, features: list):
        """
        Initialize the StandardScaling strategy.

        Args:
            features (list): The list of features to scale.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
class MinMaxScaling(FeatureEngineeringStrategy):
    """
    Strategy for applying Min-Max scaling.

    This strategy scales features to a specified range, typically [0, 1].
    """

    def __init__(self, features: list, feature_range: tuple = (0, 1)):
        """
        Initialize the MinMaxScaling strategy.

        Args:
            features (list): The list of features to scale.
            feature_range (tuple): The target range for scaling. Defaults to (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    """
    Strategy for applying One-Hot Encoding.

    This strategy converts categorical variables into binary vectors.
    """

    def __init__(self, features: list):
        """
        Initialize the OneHotEncoding strategy.

        Args:
            features (list): The list of categorical features to encode.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Engineering
class FeatureEngineer:
    """
    Context class for executing feature engineering strategies.
    """

    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initialize the FeatureEngineer with a specific strategy.

        Args:
            strategy (FeatureEngineeringStrategy): The strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Args:
            strategy (FeatureEngineeringStrategy): The new strategy to use.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass
