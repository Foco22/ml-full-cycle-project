"""
XGBoost model builder for model_balanced.

This module handles XGBoost model configuration and instantiation
using the exact hyperparameters from the notebook.
"""

import xgboost as xgb
from typing import Dict, Optional
import yaml
from pathlib import Path


def get_balanced_params() -> Dict:
    """
    Returns the exact hyperparameters for model_balanced from the notebook.

    These parameters were tuned for best balance between performance
    and generalization:
    - Test RMSE: 7.05 CLP
    - Test R²: 0.9055
    - Overfitting gap (R²): 0.0944

    Returns:
        Dictionary of XGBoost hyperparameters
    """
    return {
        # Tree structure
        'max_depth': 5,
        'min_child_weight': 4,

        # Regularization (balanced approach)
        'reg_alpha': 0.5,          # L1 regularization
        'reg_lambda': 1.5,         # L2 regularization
        'gamma': 0.15,             # Minimum loss reduction for split

        # Sampling (prevent overfitting)
        'subsample': 0.75,         # 75% row sampling per tree
        'colsample_bytree': 0.75,  # 75% column sampling per tree

        # Learning parameters
        'learning_rate': 0.04,
        'n_estimators': 250,

        # Objective and system
        'objective': 'reg:squarederror',
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1
    }


def load_params_from_config(config_path: str = 'config/model_config.yaml') -> Dict:
    """
    Load hyperparameters from configuration file.

    Args:
        config_path: Path to model configuration YAML file

    Returns:
        Dictionary of hyperparameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If hyperparameters section missing
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'hyperparameters' not in config:
        raise KeyError("'hyperparameters' section not found in config file")

    return config['hyperparameters']


def build_xgboost_model(
    params: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> xgb.XGBRegressor:
    """
    Build XGBoost regressor model with specified parameters.

    Priority:
    1. If params dict provided, use that
    2. If config_path provided, load from config file
    3. Otherwise, use default balanced parameters

    Args:
        params: Optional dictionary of hyperparameters
        config_path: Optional path to config file

    Returns:
        Configured XGBRegressor instance

    Examples:
        # Use default balanced parameters
        >>> model = build_xgboost_model()

        # Use custom parameters
        >>> custom_params = {'max_depth': 3, 'learning_rate': 0.1}
        >>> model = build_xgboost_model(params=custom_params)

        # Load from config file
        >>> model = build_xgboost_model(config_path='config/model_config.yaml')
    """
    # Determine which parameters to use
    if params is not None:
        hyperparams = params
    elif config_path is not None:
        hyperparams = load_params_from_config(config_path)
    else:
        hyperparams = get_balanced_params()

    # Remove early_stopping_rounds from params (it's a fit parameter, not init)
    hyperparams_copy = hyperparams.copy()
    hyperparams_copy.pop('early_stopping_rounds', None)

    # Create and return model
    model = xgb.XGBRegressor(**hyperparams_copy)

    return model


def get_early_stopping_rounds(
    params: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> int:
    """
    Get early stopping rounds parameter.

    Args:
        params: Optional dictionary of hyperparameters
        config_path: Optional path to config file

    Returns:
        Number of early stopping rounds (default: 20)
    """
    if params is not None and 'early_stopping_rounds' in params:
        return params['early_stopping_rounds']

    if config_path is not None:
        config_params = load_params_from_config(config_path)
        return config_params.get('early_stopping_rounds', 20)

    # Default from balanced params
    return 20


def get_model_info(model: xgb.XGBRegressor) -> Dict:
    """
    Extract model information and parameters.

    Args:
        model: Trained XGBoost model

    Returns:
        Dictionary with model information
    """
    return {
        'model_type': 'XGBoost',
        'task': 'regression',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'objective': model.objective,
        'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        'feature_names': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
    }


def validate_params(params: Dict) -> bool:
    """
    Validate hyperparameters.

    Args:
        params: Dictionary of hyperparameters

    Returns:
        True if valid

    Raises:
        ValueError: If parameters are invalid
    """
    required_params = ['objective', 'max_depth', 'learning_rate', 'n_estimators']

    # Check required parameters
    missing = [p for p in required_params if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Validate ranges
    if params['max_depth'] < 1:
        raise ValueError("max_depth must be >= 1")

    if not 0 < params['learning_rate'] <= 1:
        raise ValueError("learning_rate must be in (0, 1]")

    if params['n_estimators'] < 1:
        raise ValueError("n_estimators must be >= 1")

    # Validate objective for regression
    if params['objective'] != 'reg:squarederror':
        raise ValueError(f"Expected objective 'reg:squarederror', got '{params['objective']}'")

    return True


def create_model_with_validation(
    params: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> xgb.XGBRegressor:
    """
    Build model with parameter validation.

    Args:
        params: Optional dictionary of hyperparameters
        config_path: Optional path to config file

    Returns:
        Validated XGBRegressor instance

    Raises:
        ValueError: If parameters are invalid
    """
    # Load parameters
    if params is not None:
        hyperparams = params
    elif config_path is not None:
        hyperparams = load_params_from_config(config_path)
    else:
        hyperparams = get_balanced_params()

    # Validate parameters
    validate_params(hyperparams)

    # Build model
    return build_xgboost_model(params=hyperparams)


# Export main functions
__all__ = [
    'get_balanced_params',
    'load_params_from_config',
    'build_xgboost_model',
    'get_early_stopping_rounds',
    'get_model_info',
    'validate_params',
    'create_model_with_validation'
]