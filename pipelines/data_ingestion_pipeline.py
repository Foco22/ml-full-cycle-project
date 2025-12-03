"""
Generic Data Ingestion Pipeline
Fetches data from any source and loads it into BigQuery
Supports multiple data sources and can be easily extended
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
import logging

from src.ingestion.api_data_fetcher import BaseAPIFetcher, CMFChileAPIFetcher
from src.ingestion.data_loader import DataLoader
from src.ingestion.bigquery_loader import BigQueryLoader
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


def run_pipeline(
    source_type: str,
    config_path: str = "config/config.yaml",
    secrets_path: str = "config/secrets.yaml",
    mode: str = "incremental",
    **kwargs
):
    """
    Run the generic data ingestion pipeline

    Args:
        source_type: Type of data source ('api', 'sql', 'gcs', 'local')
        config_path: Path to configuration file
        secrets_path: Path to secrets file
        mode: Pipeline mode ('incremental', 'full', 'backfill')
        **kwargs: Additional arguments specific to source type
    """
    # Setup logging
    logger = setup_logger("data_ingestion_pipeline")
    logger.info("="*60)
    logger.info("Starting Data Ingestion Pipeline")
    logger.info(f"Source Type: {source_type}")
    logger.info(f"Mode: {mode}")
    logger.info("="*60)

    try:
        # Load configuration
        config = load_config(config_path)
        secrets = load_config(secrets_path)

        # Get pipeline config
        pipeline_config = config.get('pipeline', {})
        dataset_config = pipeline_config.get('dataset', {})

        dataset_id = dataset_config.get('dataset_id', 'data_ingestion')
        table_id = dataset_config.get('table_id', 'raw_data')

        # Initialize BigQuery loader
        project_id = config['gcp']['project_id']
        bq_loader = BigQueryLoader(project_id, dataset_id)

        # Create dataset if it doesn't exist
        logger.info("Setting up BigQuery dataset")
        bq_loader.create_dataset(location=config['gcp']['region'])

        # Fetch data based on source type
        df = None

        if source_type == 'api':
            df = fetch_from_api(config, secrets, mode, logger, **kwargs)

        elif source_type == 'sql':
            df = fetch_from_sql(config, secrets, logger, **kwargs)

        elif source_type == 'gcs':
            df = fetch_from_gcs(config, secrets, logger, **kwargs)

        elif source_type == 'local':
            df = fetch_from_local(config, secrets, logger, **kwargs)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        if df is None or df.empty:
            logger.warning("No data fetched. Pipeline completed with no data to load.")
            return

        logger.info(f"Fetched {len(df)} records")

        # Create table with dynamic schema
        logger.info("Setting up BigQuery table")
        bq_loader.create_table_from_dataframe(
            table_id=table_id,
            df_sample=df,
            description=f"Data from {source_type}",
            partition_field=dataset_config.get('partition_field'),
            cluster_fields=dataset_config.get('cluster_fields')
        )

        # Load data into BigQuery
        merge_key = dataset_config.get('merge_key')

        if mode == "full":
            # Truncate and reload
            logger.info("Loading data with WRITE_TRUNCATE mode")
            bq_loader.load_dataframe(
                df,
                table_id=table_id,
                write_disposition="WRITE_TRUNCATE"
            )
        elif merge_key:
            # Upsert to handle updates and avoid duplicates
            logger.info(f"Upserting data into BigQuery (merge key: {merge_key})")
            bq_loader.upsert_dataframe(
                df,
                table_id=table_id,
                merge_key=merge_key
            )
        else:
            # Append mode
            logger.info("Appending data to BigQuery")
            bq_loader.load_dataframe(
                df,
                table_id=table_id,
                write_disposition="WRITE_APPEND"
            )

        # Query and display sample data
        logger.info("Sample of loaded data:")
        sample_df = bq_loader.query_table(table_id, limit=5)
        logger.info(f"\n{sample_df.to_string()}")

        logger.info("="*60)
        logger.info("Data Ingestion Pipeline Completed Successfully")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


def fetch_from_api(config, secrets, mode, logger, **kwargs):
    """Fetch data from API source"""
    api_config = config.get('api', {})
    api_type = api_config.get('type', 'cmf_chile')

    api_key = secrets.get('api', {}).get('api_key')
    if not api_key:
        raise ValueError("API key not found in secrets")

    # Determine date range based on mode
    backfill_days = kwargs.get('backfill_days', 7)

    if mode == "incremental":
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        logger.info(f"Incremental mode: fetching data from {start_date.date()} to {end_date.date()}")

    elif mode == "backfill":
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backfill_days)
        logger.info(f"Backfill mode: fetching {backfill_days} days of data")

    elif mode == "full":
        start_date = datetime(1990, 1, 1)
        end_date = datetime.now()
        logger.info(f"Full mode: fetching all data from {start_date.date()}")

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Initialize appropriate fetcher
    if api_type == 'cmf_chile':
        fetcher = CMFChileAPIFetcher(api_key=api_key, config=api_config)
        currencies = api_config.get('currencies', ['usd', 'eur', 'uf'])

        logger.info(f"Fetching data from CMF Chile API: {currencies}")
        df = fetcher.fetch_date_range(start_date, end_date, currencies=currencies)

        if not df.empty:
            df = fetcher.prepare_for_bigquery(df)

    else:
        raise ValueError(f"Unsupported API type: {api_type}")

    return df


def fetch_from_sql(config, secrets, logger, **kwargs):
    """Fetch data from SQL source"""
    query = kwargs.get('query')
    if not query:
        raise ValueError("SQL query is required for SQL source")

    environment = kwargs.get('environment', 'dev')

    data_loader = DataLoader(config, secrets)
    logger.info(f"Fetching data from SQL ({environment})")
    df = data_loader.load_from_sql(query, environment)

    # Add metadata
    df['ingestion_timestamp'] = datetime.now()
    df['data_source'] = f'SQL_{environment}'

    return df


def fetch_from_gcs(config, secrets, logger, **kwargs):
    """Fetch data from Google Cloud Storage"""
    blob_path = kwargs.get('blob_path')
    if not blob_path:
        raise ValueError("blob_path is required for GCS source")

    file_format = kwargs.get('file_format', 'csv')

    data_loader = DataLoader(config, secrets)
    logger.info(f"Fetching data from GCS: {blob_path}")
    df = data_loader.load_from_gcs(blob_path, file_format)

    # Add metadata
    df['ingestion_timestamp'] = datetime.now()
    df['data_source'] = f'GCS_{blob_path}'

    return df


def fetch_from_local(config, secrets, logger, **kwargs):
    """Fetch data from local file"""
    file_path = kwargs.get('file_path')
    if not file_path:
        raise ValueError("file_path is required for local source")

    file_format = kwargs.get('file_format', 'csv')

    data_loader = DataLoader(config, secrets)
    logger.info(f"Fetching data from local file: {file_path}")
    df = data_loader.load_from_local(file_path, file_format)

    # Add metadata
    df['ingestion_timestamp'] = datetime.now()
    df['data_source'] = f'LOCAL_{file_path}'

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generic data ingestion pipeline")
    parser.add_argument("--source", required=True,
                       choices=["api", "sql", "gcs", "local"],
                       help="Data source type")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--secrets", default="config/secrets.yaml",
                       help="Path to secrets file")
    parser.add_argument("--mode", default="incremental",
                       choices=["incremental", "full", "backfill"],
                       help="Pipeline mode")
    parser.add_argument("--backfill-days", type=int, default=7,
                       help="Number of days to backfill (for backfill mode)")

    # Source-specific arguments
    parser.add_argument("--query", help="SQL query (for SQL source)")
    parser.add_argument("--blob-path", help="GCS blob path (for GCS source)")
    parser.add_argument("--file-path", help="Local file path (for local source)")
    parser.add_argument("--file-format", default="csv",
                       help="File format (csv, parquet, json)")
    parser.add_argument("--environment", default="dev",
                       choices=["dev", "prod"],
                       help="SQL environment (for SQL source)")

    args = parser.parse_args()

    # Prepare kwargs based on source type
    kwargs = {
        'backfill_days': args.backfill_days,
        'query': args.query,
        'blob_path': args.blob_path,
        'file_path': args.file_path,
        'file_format': args.file_format,
        'environment': args.environment
    }

    run_pipeline(
        source_type=args.source,
        config_path=args.config,
        secrets_path=args.secrets,
        mode=args.mode,
        **kwargs
    )
