"""
Data Ingestion Module for ML Pipeline
Handles data loading from various sources (SQL, GCS, local files)
"""

import os
import pandas as pd
import pyodbc
from google.cloud import storage, bigquery
from typing import Optional, Dict, Any
import logging
from datetime import datetime


class DataLoader:
    """Class to handle data ingestion from multiple sources"""

    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any]):
        """
        Initialize DataLoader with configuration

        Args:
            config: Configuration dictionary
            secrets: Secrets dictionary containing passwords and API keys
        """
        self.config = config
        self.secrets = secrets
        self.logger = logging.getLogger(__name__)

    def load_from_sql(self, query: str, environment: str = "dev") -> pd.DataFrame:
        """
        Load data from SQL Server / Azure Synapse

        Args:
            query: SQL query to execute
            environment: 'prod' or 'dev' environment

        Returns:
            DataFrame with query results
        """
        try:
            sql_config = self.config['sql']

            # Select server and password based on environment
            server = sql_config[f'server_{environment}']
            password = self.secrets['sql'][f'password_{environment}']
            schema = sql_config.get(f'schema_{environment}', '')

            # Build connection string
            conn_str = (
                f"DRIVER={sql_config['driver']};"
                f"SERVER={server};"
                f"DATABASE={sql_config['database']};"
                f"UID={sql_config['username']};"
                f"PWD={password}"
            )

            self.logger.info(f"Connecting to SQL Server ({environment})")

            # Connect and execute query
            with pyodbc.connect(conn_str) as conn:
                if schema:
                    query = f"USE {schema}; {query}"
                df = pd.read_sql(query, conn)

            self.logger.info(f"Loaded {len(df)} rows from SQL")
            return df

        except Exception as e:
            self.logger.error(f"Error loading data from SQL: {str(e)}")
            raise

    def load_from_gcs(self, blob_path: str, file_format: str = "csv") -> pd.DataFrame:
        """
        Load data from Google Cloud Storage

        Args:
            blob_path: Path to blob in format 'bucket/path/to/file.csv'
            file_format: File format ('csv', 'parquet', 'json')

        Returns:
            DataFrame with data
        """
        try:
            # Initialize GCS client
            storage_client = storage.Client(project=self.config['gcp']['project_id'])

            # Parse bucket and blob path
            parts = blob_path.split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            self.logger.info(f"Loading data from GCS: gs://{blob_path}")

            # Download to temporary location
            temp_file = f"/tmp/temp_data_{datetime.now().timestamp()}.{file_format}"
            blob.download_to_filename(temp_file)

            # Load based on format
            if file_format == "csv":
                df = pd.read_csv(temp_file)
            elif file_format == "parquet":
                df = pd.read_parquet(temp_file)
            elif file_format == "json":
                df = pd.read_json(temp_file)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Clean up temp file
            os.remove(temp_file)

            self.logger.info(f"Loaded {len(df)} rows from GCS")
            return df

        except Exception as e:
            self.logger.error(f"Error loading data from GCS: {str(e)}")
            raise

    def load_from_bigquery(self, query: str) -> pd.DataFrame:
        """
        Load data from BigQuery

        Args:
            query: BigQuery SQL query

        Returns:
            DataFrame with query results
        """
        try:
            client = bigquery.Client(project=self.config['gcp']['project_id'])

            self.logger.info("Executing BigQuery query")
            df = client.query(query).to_dataframe()

            self.logger.info(f"Loaded {len(df)} rows from BigQuery")
            return df

        except Exception as e:
            self.logger.error(f"Error loading data from BigQuery: {str(e)}")
            raise

    def save_to_gcs(self, df: pd.DataFrame, blob_path: str, file_format: str = "csv"):
        """
        Save DataFrame to Google Cloud Storage

        Args:
            df: DataFrame to save
            blob_path: Path to blob in format 'bucket/path/to/file.csv'
            file_format: File format ('csv', 'parquet', 'json')
        """
        try:
            storage_client = storage.Client(project=self.config['gcp']['project_id'])

            parts = blob_path.split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Save to temp file first
            temp_file = f"/tmp/temp_data_{datetime.now().timestamp()}.{file_format}"

            if file_format == "csv":
                df.to_csv(temp_file, index=False)
            elif file_format == "parquet":
                df.to_parquet(temp_file, index=False)
            elif file_format == "json":
                df.to_json(temp_file, orient='records')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Upload to GCS
            blob.upload_from_filename(temp_file)
            os.remove(temp_file)

            self.logger.info(f"Saved {len(df)} rows to GCS: gs://{blob_path}")

        except Exception as e:
            self.logger.error(f"Error saving data to GCS: {str(e)}")
            raise

