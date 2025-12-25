"""
Feature engineering and selection for model_balanced.

This module implements the exact feature engineering pipeline from the notebook
to ensure consistency between training and inference.

Features created:
- Lag features (10 lags: 1, 2, 3, 7, 14, 30, 60, 90, 180, 365 days)
- Rolling statistics (7 windows × 4 metrics = 28 features)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Ratio features (price vs moving averages)

Model_balanced uses 13 of these features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import yaml
from pathlib import Path


def get_balanced_features() -> List[str]:
    """
    Returns the 13 features used by model_balanced.

    These features were selected based on importance > 0.0001 from the
    original 70-feature model.

    Returns:
        List of 13 feature names
    """
    return [
        # Lag features (3)
        'lag_1d',
        'lag_2d',
        'lag_3d',

        # Rolling mean features (2)
        'rolling_mean_7d',
        'rolling_mean_30d',

        # Rolling max features (3)
        'rolling_max_7d',
        'rolling_max_14d',
        'rolling_max_30d',

        # Rolling min features (3)
        'rolling_min_7d',
        'rolling_min_14d',
        'rolling_min_30d',

        # Technical indicators (2)
        'bollinger_lower',
        'ratio_vs_ma7'
    ]


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for a given column.

    Args:
        df: Input dataframe
        column: Column name to create lags for
        lags: List of lag periods (e.g., [1, 2, 3, 7, 14, 30])

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()

    for lag in lags:
        df[f'lag_{lag}d'] = df[column].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int]
) -> pd.DataFrame:
    """
    Create rolling window statistics.

    Args:
        df: Input dataframe
        column: Column name to compute rolling stats for
        windows: List of window sizes (e.g., [7, 14, 30, 60, 90, 180, 365])

    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()

    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}d'] = df[column].rolling(window=window).mean()

        # Rolling standard deviation (volatility)
        df[f'rolling_std_{window}d'] = df[column].rolling(window=window).std()

        # Rolling min
        df[f'rolling_min_{window}d'] = df[column].rolling(window=window).min()

        # Rolling max
        df[f'rolling_max_{window}d'] = df[column].rolling(window=window).max()

    return df


def create_bollinger_bands(
    df: pd.DataFrame,
    column: str,
    window: int = 20,
    num_std: int = 2
) -> pd.DataFrame:
    """
    Create Bollinger Bands technical indicator.

    Args:
        df: Input dataframe
        column: Column name
        window: Moving average window (default: 20)
        num_std: Number of standard deviations (default: 2)

    Returns:
        DataFrame with Bollinger Bands added
    """
    df = df.copy()

    # Middle band (SMA)
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    # Upper and lower bands
    df['bollinger_upper'] = rolling_mean + (rolling_std * num_std)
    df['bollinger_lower'] = rolling_mean - (rolling_std * num_std)
    df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']

    return df


def create_rsi(df: pd.DataFrame, column: str, window: int = 14) -> pd.DataFrame:
    """
    Create RSI (Relative Strength Index) technical indicator.

    Args:
        df: Input dataframe
        column: Column name
        window: RSI window (default: 14)

    Returns:
        DataFrame with RSI added
    """
    df = df.copy()

    # Calculate price changes
    delta = df[column].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def create_macd(
    df: pd.DataFrame,
    column: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Create MACD (Moving Average Convergence Divergence) indicator.

    Args:
        df: Input dataframe
        column: Column name
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        DataFrame with MACD features added
    """
    df = df.copy()

    # Calculate EMAs
    ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

    # MACD line
    df['macd'] = ema_fast - ema_slow

    # Signal line
    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

    # MACD histogram
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df


def create_ratio_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Create ratio features (price vs moving averages).

    Args:
        df: Input dataframe
        column: Column name

    Returns:
        DataFrame with ratio features added
    """
    df = df.copy()

    # Ratio vs different moving averages
    for window in [7, 30, 365]:
        ma = df[column].rolling(window=window).mean()
        df[f'ratio_vs_ma{window}'] = df[column] / ma

    return df


def engineer_all_features(
    df: pd.DataFrame,
    config: Dict = None
) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline.

    This creates all possible features. The actual model uses only 13 of these.

    Args:
        df: Input dataframe with 'usdclp_obs' column
        config: Optional configuration dictionary

    Returns:
        DataFrame with all engineered features
    """
    # Validate input
    if 'usdclp_obs' not in df.columns:
        raise ValueError("DataFrame must contain 'usdclp_obs' column")

    target_col = 'usdclp_obs'
    df = df.copy()

    # 1. Lag features
    lags = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365]
    df = create_lag_features(df, target_col, lags)

    # 2. Rolling window features
    windows = [7, 14, 30, 60, 90, 180, 365]
    df = create_rolling_features(df, target_col, windows)

    # 3. Bollinger Bands
    df = create_bollinger_bands(df, target_col, window=20, num_std=2)

    # 4. RSI
    df = create_rsi(df, target_col, window=14)

    # 5. MACD
    df = create_macd(df, target_col)

    # 6. Ratio features
    df = create_ratio_features(df, target_col)

    return df


def select_balanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the 13 features used by model_balanced.

    Args:
        df: DataFrame with all engineered features

    Returns:
        DataFrame with only the 13 selected features
    """
    selected_features = get_balanced_features()

    # Validate all features exist
    missing_features = set(selected_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    return df[selected_features]


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = 'usdclp_obs',
    drop_na: bool = True
) -> tuple:
    """
    Complete pipeline: engineer features and prepare X, y for training.

    Args:
        df: Input dataframe with 'usdclp_obs' column
        target_column: Name of target variable
        drop_na: Whether to drop rows with NaN values

    Returns:
        Tuple of (X, y, feature_names) where:
        - X: Feature matrix (13 features)
        - y: Target variable
        - feature_names: List of feature names
    """
    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Store target before feature engineering
    y = df[target_column].copy()

    # Engineer all features
    df_engineered = engineer_all_features(df)

    # Select balanced features (13 features)
    X = select_balanced_features(df_engineered)

    # Drop rows with NaN values (from lag features)
    if drop_na:
        # Find rows where both X and y are not NaN
        valid_idx = X.notna().all(axis=1) & y.notna()

        X = X[valid_idx]
        y = y[valid_idx]

    feature_names = X.columns.tolist()

    return X, y, feature_names


def load_feature_config(config_path: str = 'config/model_config.yaml') -> Dict:
    """
    Load feature configuration from YAML file.

    Args:
        config_path: Path to model configuration file

    Returns:
        Feature configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('features', {})


def validate_features(df: pd.DataFrame, expected_features: List[str]) -> bool:
    """
    Validate that dataframe contains all expected features.

    Args:
        df: Input dataframe
        expected_features: List of expected feature names

    Returns:
        True if all features present, raises ValueError otherwise
    """
    missing = set(expected_features) - set(df.columns)

    if missing:
        raise ValueError(f"Missing features: {missing}")

    return True


# Export main functions
__all__ = [
    'get_balanced_features',
    'engineer_all_features',
    'select_balanced_features',
    'prepare_training_data',
    'create_lag_features',
    'create_rolling_features',
    'create_bollinger_bands',
    'create_rsi',
    'create_macd',
    'create_ratio_features',
    'load_feature_config',
    'validate_features'
]