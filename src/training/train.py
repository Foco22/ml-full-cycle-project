"""
Training script for model_balanced with Vertex AI and GCS integration.

This script handles:
- Data loading from BigQuery
- Feature engineering
- Model training with XGBoost
- Model validation
- Saving artifacts to GCS bucket
- MLOps best practices (versioning, metadata, logging)

Usage:
    # Local training
    python -m src.training.train --config config/training_config.yaml --local

    # Vertex AI training
    python -m src.training.train --config config/training_config.yaml
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import gzip
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from google.cloud import bigquery, storage

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.feature_selector import prepare_training_data, get_balanced_features
from src.training.model_builder import build_xgboost_model, get_early_stopping_rounds
from src.training.model_validator import (
    validate_model,
    check_performance_gates,
    generate_validation_report,
    get_feature_importance
)
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.gcs_utils import upload_to_gcs, list_gcs_files

logger = get_logger(__name__)


def load_data_from_bigquery(config: Dict) -> pd.DataFrame:
    """
    Load data from BigQuery.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with raw data
    """
    logger.info("Loading data from BigQuery...")

    bq_config = config['data']['bigquery']
    project_id = os.getenv('GCP_PROJECT_ID', bq_config['project_id'])

    client = bigquery.Client(project=project_id)

    # Build query
    table_ref = f"{project_id}.{bq_config['dataset']}.{bq_config['table']}"

    query = f"SELECT * FROM `{table_ref}`"

    # Add date range filter if specified
    date_range = bq_config.get('date_range', {})
    if date_range.get('start'):
        query += f" WHERE {bq_config['date_column']} >= '{date_range['start']}'"
        if date_range.get('end'):
            query += f" AND {bq_config['date_column']} <= '{date_range['end']}'"
    elif date_range.get('end'):
        query += f" WHERE {bq_config['date_column']} <= '{date_range['end']}'"

    query += f" ORDER BY {bq_config['date_column']}"

    logger.info(f"Query: {query}")

    # Execute query
    df = client.query(query).to_dataframe()

    logger.info(f"Loaded {len(df):,} rows from BigQuery")

    # Data quality checks
    if len(df) > 0:
        # Show columns first
        logger.info(f"Columns in data: {list(df.columns)}")

        # Check date column
        date_col = bq_config['date_column']
        if date_col in df.columns:
            logger.info(f"Date range ({date_col}): {df[date_col].min()} to {df[date_col].max()}")
        else:
            logger.warning(f"Date column '{date_col}' not found! Available columns: {list(df.columns)}")

        # Check target column
        target_col = bq_config.get('target_column', 'usdclp_obs')
        if target_col in df.columns:
            logger.info(f"{target_col} statistics:")
            logger.info(f"  Min: {df[target_col].min():.2f}")
            logger.info(f"  Max: {df[target_col].max():.2f}")
            logger.info(f"  Mean: {df[target_col].mean():.2f}")
            logger.info(f"  Std: {df[target_col].std():.2f}")
            logger.info(f"  Missing: {df[target_col].isna().sum()}")
        else:
            logger.warning(f"Target column '{target_col}' not found! Available columns: {list(df.columns)}")
    else:
        logger.error("No data loaded from BigQuery!")

    return df


def load_training_data(config: Dict) -> pd.DataFrame:
    """
    Load training data from BigQuery.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with raw data
    """
    return load_data_from_bigquery(config)


def prepare_data(
    df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for training: feature engineering and train/test split.

    Args:
        df: Raw dataframe
        config: Configuration dictionary

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing data...")

    # Get target column name
    target_col = config['data'].get('bigquery', {}).get('target_column', 'usdclp_obs')

    # Feature engineering
    logger.info("Engineering features...")
    X, y, feature_names = prepare_training_data(df, target_column=target_col)

    logger.info(f"Feature engineering complete: {len(feature_names)} features")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Samples after dropping NaN: {len(X):,}")

    # Train/test split
    split_config = config['model']['train_test_split']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_config['test_size'],
        shuffle=split_config['shuffle'],
        random_state=split_config['random_state']
    )

    logger.info(f"Train samples: {len(X_train):,}")
    logger.info(f"Test samples: {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def train_model(config: Dict) -> Tuple:
    """
    Train XGBoost model.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info("="*80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("="*80)

    # Load data
    df = load_training_data(config)

    # Prepare features
    X_train, X_test, y_train, y_test = prepare_data(df, config)
    feature_names = X_train.columns.tolist()

    # Build model
    logger.info("Building XGBoost model...")
    model_config_path = config['model']['config_path']
    model = build_xgboost_model(config_path=model_config_path)

    # Get early stopping rounds
    early_stopping = get_early_stopping_rounds(config_path=model_config_path)

    # Train model
    logger.info("Training model...")
    logger.info(f"Early stopping rounds: {early_stopping}")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=config.get('training', {}).get('verbose', True)
    )

    logger.info("Training complete!")

    return model, X_train, X_test, y_train, y_test, feature_names


def save_model_artifacts(
    model,
    feature_names: list,
    metrics: Dict,
    config: Dict,
    version: str,
    local: bool = False
) -> Dict[str, str]:
    """
    Save model artifacts (model, features, metadata) to GCS or local.

    Args:
        model: Trained model
        feature_names: List of feature names
        metrics: Validation metrics
        config: Configuration dictionary
        version: Model version
        local: If True, save locally instead of GCS

    Returns:
        Dictionary of saved artifact paths
    """
    logger.info("Saving model artifacts...")

    paths = {}

    # Determine output location
    if local:
        output_config = config['output']['local']
        base_dir = Path(output_config['model_dir']) / f'model_balanced_{version}'
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = base_dir / 'model.pkl'
        with gzip.open(model_path, 'wb') as f:
            pickle.dump(model, f)
        paths['model'] = str(model_path)

        # Save features
        features_path = base_dir / 'features.json'
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        paths['features'] = str(features_path)

        # Save metadata
        metadata_path = base_dir / 'metadata.json'
        metadata = create_metadata(model, feature_names, metrics, config, version)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        paths['metadata'] = str(metadata_path)

        logger.info(f"Artifacts saved locally to: {base_dir}")

    else:
        # Save to GCS
        gcs_config = config['output']['gcs']

        # Replace {version} placeholder
        model_path_template = gcs_config['model_path'].replace('{version}', version)
        features_path_template = gcs_config['features_path'].replace('{version}', version)
        metadata_path_template = gcs_config['metadata_path'].replace('{version}', version)

        # Create temporary files
        temp_dir = Path('/tmp/model_artifacts')
        temp_dir.mkdir(exist_ok=True)

        # Save model
        temp_model = temp_dir / 'model.pkl.gz'
        with gzip.open(temp_model, 'wb') as f:
            pickle.dump(model, f)
        upload_to_gcs(str(temp_model), model_path_template)
        paths['model'] = model_path_template

        # Save features
        temp_features = temp_dir / 'features.json'
        with open(temp_features, 'w') as f:
            json.dump(feature_names, f, indent=2)
        upload_to_gcs(str(temp_features), features_path_template)
        paths['features'] = features_path_template

        # Save metadata
        temp_metadata = temp_dir / 'metadata.json'
        metadata = create_metadata(model, feature_names, metrics, config, version)
        with open(temp_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
        upload_to_gcs(str(temp_metadata), metadata_path_template)
        paths['metadata'] = metadata_path_template

        logger.info(f"Artifacts uploaded to GCS bucket")

        # Clean up temp files
        if config.get('cleanup', {}).get('delete_temp_files', True):
            import shutil
            shutil.rmtree(temp_dir)

    return paths


def save_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict,
    version: str,
    local: bool = False
) -> Dict[str, str]:
    """
    Save training and test datasets to GCS or local (for reproducibility).

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Configuration dictionary
        version: Model version
        local: If True, save locally

    Returns:
        Dictionary of saved data paths
    """
    logger.info("Saving training datasets...")

    paths = {}

    # Combine X and y
    train_df = X_train.copy()
    train_df['target'] = y_train.values

    test_df = X_test.copy()
    test_df['target'] = y_test.values

    if local:
        data_dir = Path('data') / f'training_{version}'
        data_dir.mkdir(parents=True, exist_ok=True)

        train_path = data_dir / 'train_data.csv'
        test_path = data_dir / 'test_data.csv'

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        paths['train'] = str(train_path)
        paths['test'] = str(test_path)

        logger.info(f"Data saved locally to: {data_dir}")

    else:
        gcs_config = config['output']['gcs']

        train_path_template = gcs_config['train_data_path'].replace('{version}', version)
        test_path_template = gcs_config['test_data_path'].replace('{version}', version)

        # Save to temp then upload
        temp_train = '/tmp/train_data.csv'
        temp_test = '/tmp/test_data.csv'

        train_df.to_csv(temp_train, index=False)
        test_df.to_csv(temp_test, index=False)

        upload_to_gcs(temp_train, train_path_template)
        upload_to_gcs(temp_test, test_path_template)

        paths['train'] = train_path_template
        paths['test'] = test_path_template

        logger.info("Training data uploaded to GCS")

        # Clean up
        os.remove(temp_train)
        os.remove(temp_test)

    return paths


def create_metadata(
    model,
    feature_names: list,
    metrics: Dict,
    config: Dict,
    version: str
) -> Dict:
    """
    Create model metadata for tracking.

    Args:
        model: Trained model
        feature_names: List of features
        metrics: Validation metrics
        config: Configuration
        version: Model version

    Returns:
        Metadata dictionary
    """
    # Get git commit if available
    git_commit = os.getenv('GIT_COMMIT', 'unknown')
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        pass

    metadata = {
        'model_name': 'model_balanced',
        'version': version,
        'created_at': datetime.utcnow().isoformat(),
        'features': {
            'count': len(feature_names),
            'names': feature_names
        },
        'metrics': metrics,
        'hyperparameters': {
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'n_estimators': model.n_estimators,
            'reg_alpha': model.reg_alpha,
            'reg_lambda': model.reg_lambda,
            'subsample': model.subsample,
            'colsample_bytree': model.colsample_bytree
        },
        'training': {
            'data_source': config['data']['source'],
            'test_size': config['model']['train_test_split']['test_size']
        },
        'environment': {
            'python_version': sys.version,
            'git_commit': git_commit,
            'user': os.getenv('USER', 'unknown')
        }
    }

    return metadata


def generate_version(strategy: str = 'timestamp') -> str:
    """
    Generate model version string.

    Args:
        strategy: Versioning strategy (timestamp, semantic, auto_increment)

    Returns:
        Version string
    """
    if strategy == 'timestamp':
        return datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    elif strategy == 'semantic':
        # Would need to track this externally
        return 'v1.0.0'
    else:
        return 'v1'


def main(config_path: str, local: bool = False):
    """
    Main training pipeline.

    Args:
        config_path: Path to training configuration
        local: If True, save artifacts locally instead of GCS
    """
    start_time = datetime.utcnow()

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)

        # Generate version
        version = generate_version(config['output']['versioning']['strategy'])
        logger.info(f"Model version: {version}")

        # Train model
        model, X_train, X_test, y_train, y_test, feature_names = train_model(config)

        # Validate model
        logger.info("Validating model...")
        validation_results = validate_model(model, X_train, y_train, X_test, y_test)

        # Check performance gates
        gates_config = config.get('validation', {}).get('performance_gates', {})
        if gates_config.get('enabled', True):
            gates = check_performance_gates(
                validation_results['test'],
                min_r2=gates_config.get('min_test_r2', 0.85),
                max_rmse=gates_config.get('max_test_rmse', 10.0)
            )

            if not gates['passed']:
                logger.error("Performance gates failed!")
                logger.error(gates['message'])
                for gate_name, gate_result in gates['gates'].items():
                    logger.error(f"  {gate_result['message']}")

                raise ValueError("Model did not meet performance requirements")

        # Generate report
        report = generate_validation_report(model, validation_results, feature_names)
        logger.info("\n" + report)

        # Save model artifacts
        artifact_paths = save_model_artifacts(
            model,
            feature_names,
            validation_results,
            config,
            version,
            local
        )

        # Save training data
        data_paths = save_training_data(
            X_train, y_train,
            X_test, y_test,
            config,
            version,
            local
        )

        # Training complete
        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Version: {version}")
        logger.info(f"Test RMSE: {validation_results['test']['rmse']:.2f} CLP")
        logger.info(f"Test R²: {validation_results['test']['r2']:.4f}")
        logger.info(f"Model path: {artifact_paths['model']}")
        logger.info("="*80)

        return {
            'version': version,
            'metrics': validation_results,
            'paths': {**artifact_paths, **data_paths}
        }

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model_balanced with Vertex AI')
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Save artifacts locally instead of GCS'
    )

    args = parser.parse_args()

    main(args.config, args.local)