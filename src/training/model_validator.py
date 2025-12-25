"""
Model validation and metrics calculation for model_balanced.

This module provides functions to:
- Calculate regression metrics (RMSE, R², MAE)
- Validate model performance against baselines
- Check for overfitting
- Generate validation reports
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path
import xgboost as xgb


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics:
        - rmse: Root Mean Squared Error
        - r2: R-squared score
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - mape: Mean Absolute Percentage Error
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'rmse': float(rmse),
        'r2': float(r2),
        'mae': float(mae),
        'mse': float(mse),
        'mape': float(mape)
    }


def validate_model(
    model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Validate model performance on train and test sets.

    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with train/test metrics and overfitting analysis
    """
    # Generate predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics for both sets
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    # Check for overfitting
    overfitting_analysis = check_overfitting(train_metrics, test_metrics)

    return {
        'train': train_metrics,
        'test': test_metrics,
        'overfitting': overfitting_analysis,
        'samples': {
            'train': len(y_train),
            'test': len(y_test)
        }
    }


def check_overfitting(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    rmse_threshold: float = 10.0,
    r2_threshold: float = 0.15
) -> Dict:
    """
    Check if model is overfitting.

    Overfitting indicators:
    - Test RMSE significantly higher than train RMSE
    - Test R² significantly lower than train R²

    Args:
        train_metrics: Training set metrics
        test_metrics: Test set metrics
        rmse_threshold: Max acceptable RMSE difference (default: 10.0)
        r2_threshold: Max acceptable R² difference (default: 0.15)

    Returns:
        Dictionary with overfitting analysis
    """
    # Calculate differences
    rmse_diff = test_metrics['rmse'] - train_metrics['rmse']
    r2_diff = train_metrics['r2'] - test_metrics['r2']

    # Determine if overfitting
    is_overfitting_rmse = rmse_diff > rmse_threshold
    is_overfitting_r2 = r2_diff > r2_threshold
    is_overfitting = is_overfitting_rmse or is_overfitting_r2

    # Overall status
    if not is_overfitting:
        status = 'good'
        message = 'Model generalizes well'
    elif is_overfitting_rmse and is_overfitting_r2:
        status = 'severe'
        message = 'Severe overfitting detected on both RMSE and R²'
    else:
        status = 'moderate'
        message = 'Moderate overfitting detected'

    return {
        'is_overfitting': is_overfitting,
        'status': status,
        'message': message,
        'rmse_difference': float(rmse_diff),
        'r2_difference': float(r2_diff),
        'thresholds': {
            'rmse_threshold': rmse_threshold,
            'r2_threshold': r2_threshold
        },
        'flags': {
            'rmse_overfitting': is_overfitting_rmse,
            'r2_overfitting': is_overfitting_r2
        }
    }


def validate_against_baseline(
    metrics: Dict[str, float],
    baseline_path: str = 'config/model_config.yaml'
) -> Dict:
    """
    Compare model performance against baseline metrics from notebook.

    Args:
        metrics: Current model metrics
        baseline_path: Path to config file with baseline metrics

    Returns:
        Dictionary with comparison results
    """
    # Load baseline metrics from config
    baseline = load_baseline_metrics(baseline_path)

    if not baseline:
        return {
            'comparison_available': False,
            'message': 'No baseline metrics found'
        }

    # Compare test metrics
    test_baseline = baseline.get('test', {})

    rmse_diff = metrics['rmse'] - test_baseline.get('rmse', 0)
    r2_diff = metrics['r2'] - test_baseline.get('r2', 0)

    # Determine if performance improved
    rmse_improved = rmse_diff < 0  # Lower is better
    r2_improved = r2_diff > 0      # Higher is better

    # Overall assessment
    if rmse_improved and r2_improved:
        status = 'improved'
        message = 'Model performance improved on both metrics'
    elif rmse_improved or r2_improved:
        status = 'mixed'
        message = 'Model performance improved on one metric'
    else:
        status = 'degraded'
        message = 'Model performance degraded compared to baseline'

    return {
        'comparison_available': True,
        'status': status,
        'message': message,
        'current': {
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        },
        'baseline': {
            'rmse': test_baseline.get('rmse'),
            'r2': test_baseline.get('r2')
        },
        'differences': {
            'rmse': float(rmse_diff),
            'r2': float(r2_diff)
        },
        'improvements': {
            'rmse_improved': rmse_improved,
            'r2_improved': r2_improved
        }
    }


def load_baseline_metrics(config_path: str) -> Optional[Dict]:
    """
    Load baseline metrics from configuration file.

    Args:
        config_path: Path to model config file

    Returns:
        Baseline metrics dictionary or None
    """
    config_file = Path(config_path)

    if not config_file.exists():
        return None

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('baseline_metrics')


def check_performance_gates(
    metrics: Dict[str, float],
    min_r2: float = 0.85,
    max_rmse: float = 10.0
) -> Dict:
    """
    Check if model meets minimum performance requirements.

    Performance gates prevent deploying poor-quality models.

    Args:
        metrics: Model metrics
        min_r2: Minimum acceptable R² (default: 0.85)
        max_rmse: Maximum acceptable RMSE (default: 10.0)

    Returns:
        Dictionary with gate check results
    """
    r2_passed = metrics['r2'] >= min_r2
    rmse_passed = metrics['rmse'] <= max_rmse
    all_passed = r2_passed and rmse_passed

    return {
        'passed': all_passed,
        'gates': {
            'r2': {
                'passed': r2_passed,
                'value': metrics['r2'],
                'threshold': min_r2,
                'message': f"R² {metrics['r2']:.4f} {'≥' if r2_passed else '<'} {min_r2}"
            },
            'rmse': {
                'passed': rmse_passed,
                'value': metrics['rmse'],
                'threshold': max_rmse,
                'message': f"RMSE {metrics['rmse']:.2f} {'≤' if rmse_passed else '>'} {max_rmse}"
            }
        },
        'message': 'All gates passed' if all_passed else 'Some gates failed'
    }


def generate_validation_report(
    model: xgb.XGBRegressor,
    validation_results: Dict,
    feature_names: list
) -> str:
    """
    Generate a human-readable validation report.

    Args:
        model: Trained model
        validation_results: Results from validate_model()
        feature_names: List of feature names

    Returns:
        Formatted validation report string
    """
    train = validation_results['train']
    test = validation_results['test']
    overfitting = validation_results['overfitting']
    samples = validation_results['samples']

    report = []
    report.append("=" * 80)
    report.append("MODEL VALIDATION REPORT - model_balanced")
    report.append("=" * 80)
    report.append("")

    # Model info
    report.append("MODEL CONFIGURATION:")
    report.append(f"  Model Type: XGBoost Regressor")
    report.append(f"  Features: {len(feature_names)}")
    report.append(f"  Training Samples: {samples['train']:,}")
    report.append(f"  Test Samples: {samples['test']:,}")
    report.append("")

    # Training metrics
    report.append("TRAINING SET PERFORMANCE:")
    report.append(f"  RMSE:  {train['rmse']:.2f} CLP")
    report.append(f"  R²:    {train['r2']:.4f}")
    report.append(f"  MAE:   {train['mae']:.2f} CLP")
    report.append(f"  MAPE:  {train['mape']:.2f}%")
    report.append("")

    # Test metrics
    report.append("TEST SET PERFORMANCE:")
    report.append(f"  RMSE:  {test['rmse']:.2f} CLP")
    report.append(f"  R²:    {test['r2']:.4f}")
    report.append(f"  MAE:   {test['mae']:.2f} CLP")
    report.append(f"  MAPE:  {test['mape']:.2f}%")
    report.append("")

    # Overfitting analysis
    report.append("OVERFITTING ANALYSIS:")
    report.append(f"  Status: {overfitting['status'].upper()}")
    report.append(f"  Message: {overfitting['message']}")
    report.append(f"  RMSE Difference: {overfitting['rmse_difference']:.2f} CLP")
    report.append(f"  R² Difference: {overfitting['r2_difference']:.4f}")
    report.append("")

    # Feature importance (top 10)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-10:][::-1]

        report.append("TOP 10 FEATURE IMPORTANCES:")
        for idx in top_indices:
            report.append(f"  {feature_names[idx]:25s} {importance[idx]:.4f}")
        report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def get_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: list
) -> pd.DataFrame:
    """
    Get feature importance as a sorted DataFrame.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        DataFrame with features and importances sorted by importance
    """
    importance = model.feature_importances_

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    df = df.sort_values('importance', ascending=False)
    df = df.reset_index(drop=True)

    return df


# Export main functions
__all__ = [
    'calculate_metrics',
    'validate_model',
    'check_overfitting',
    'validate_against_baseline',
    'check_performance_gates',
    'generate_validation_report',
    'get_feature_importance'
]