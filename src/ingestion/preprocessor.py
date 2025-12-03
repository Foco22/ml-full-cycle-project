"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple, Optional
import logging
import joblib
import os


class DataPreprocessor:
    """Class to handle data preprocessing and feature engineering"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataPreprocessor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.encoders = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: handle missing values, duplicates, etc.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning")

        # Remove duplicates
        df = df.drop_duplicates()
        self.logger.info(f"Removed duplicates, shape: {df.shape}")

        # Handle missing values
        # Numeric columns: fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Categorical columns: fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        self.logger.info("Data cleaning completed")
        return df

    def encode_categorical(self, df: pd.DataFrame,
                          categorical_columns: List[str],
                          method: str = "onehot") -> pd.DataFrame:
        """
        Encode categorical variables

        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            method: Encoding method ('onehot' or 'label')

        Returns:
            DataFrame with encoded features
        """
        self.logger.info(f"Encoding categorical variables using {method}")

        df_encoded = df.copy()

        for col in categorical_columns:
            if col not in df.columns:
                continue

            if method == "label":
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col])
                else:
                    df_encoded[col] = self.encoders[col].transform(df[col])

            elif method == "onehot":
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.encoders[col].fit_transform(df[[col]])
                else:
                    encoded = self.encoders[col].transform(df[[col]])

                # Create column names
                feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

                # Drop original column and add encoded columns
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

        self.logger.info(f"Encoded features shape: {df_encoded.shape}")
        return df_encoded

    def scale_features(self, df: pd.DataFrame,
                       feature_columns: List[str],
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler

        Args:
            df: Input DataFrame
            feature_columns: List of columns to scale
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            DataFrame with scaled features
        """
        self.logger.info("Scaling numerical features")

        df_scaled = df.copy()

        if fit:
            self.scaler = StandardScaler()
            df_scaled[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[feature_columns] = self.scaler.transform(df[feature_columns])

        return df_scaled

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features based on existing ones
        Customize this method based on your specific use case

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features")

        df_eng = df.copy()

        # Example feature engineering (customize as needed)
        # Add your domain-specific feature engineering here

        return df_eng

    def split_data(self, df: pd.DataFrame,
                   target_column: str,
                   test_size: Optional[float] = None,
                   random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets

        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            random_state: Random state for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = test_size or self.config['data'].get('train_test_split', 0.2)
        random_state = random_state or self.config['data'].get('random_state', 42)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def save_preprocessors(self, output_dir: str):
        """
        Save preprocessing artifacts (scaler, encoders)

        Args:
            output_dir: Directory to save artifacts
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Saved scaler to {scaler_path}")

        if self.encoders:
            encoders_path = os.path.join(output_dir, 'encoders.pkl')
            joblib.dump(self.encoders, encoders_path)
            self.logger.info(f"Saved encoders to {encoders_path}")

    def load_preprocessors(self, input_dir: str):
        """
        Load preprocessing artifacts (scaler, encoders)

        Args:
            input_dir: Directory containing artifacts
        """
        scaler_path = os.path.join(input_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Loaded scaler from {scaler_path}")

        encoders_path = os.path.join(input_dir, 'encoders.pkl')
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
            self.logger.info(f"Loaded encoders from {encoders_path}")
