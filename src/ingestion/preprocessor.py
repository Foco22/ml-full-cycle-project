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

