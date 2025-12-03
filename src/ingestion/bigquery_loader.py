"""
BigQuery Loader Module
Handles loading data into BigQuery tables
"""

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime


class BigQueryLoader:
    """Class to handle BigQuery operations"""

    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize BigQuery loader

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)

    def create_dataset(self, location: str = "US") -> bigquery.Dataset:
        """
        Create BigQuery dataset if it doesn't exist

        Args:
            location: Dataset location

        Returns:
            Dataset object
        """
        dataset_ref = f"{self.project_id}.{self.dataset_id}"

        try:
            dataset = self.client.get_dataset(dataset_ref)
            self.logger.info(f"Dataset {dataset_ref} already exists")
            return dataset
        except NotFound:
            self.logger.info(f"Creating dataset {dataset_ref}")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            dataset = self.client.create_dataset(dataset, timeout=30)
            self.logger.info(f"Created dataset {dataset_ref}")
            return dataset

    def create_table_from_dataframe(
        self,
        table_id: str,
        df_sample: pd.DataFrame,
        description: str = "Auto-generated table",
        partition_field: Optional[str] = None,
        cluster_fields: Optional[List[str]] = None
    ) -> bigquery.Table:
        """
        Create BigQuery table with schema inferred from DataFrame

        Args:
            table_id: Table ID
            df_sample: Sample DataFrame to infer schema
            description: Table description
            partition_field: Field to partition by (must be DATE or TIMESTAMP)
            cluster_fields: Fields to cluster by

        Returns:
            Table object
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        try:
            table = self.client.get_table(table_ref)
            self.logger.info(f"Table {table_ref} already exists")
            return table
        except NotFound:
            self.logger.info(f"Creating table {table_ref} with inferred schema")

            # Let BigQuery infer schema from DataFrame
            # Create a small job config to get schema
            job_config = bigquery.LoadJobConfig(
                autodetect=True,
                write_disposition="WRITE_EMPTY"
            )

            # Create empty table with inferred schema
            temp_table = bigquery.Table(table_ref)
            temp_table.description = description

            # Add partitioning if specified
            if partition_field:
                temp_table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field=partition_field
                )

            # Add clustering if specified
            if cluster_fields:
                temp_table.clustering_fields = cluster_fields

            # First create the table structure
            temp_table = self.client.create_table(temp_table)

            # Load sample data to establish schema (then delete)
            sample_data = df_sample.head(1)
            self.client.load_table_from_dataframe(
                sample_data, table_ref, job_config=job_config
            ).result()

            # Delete the sample row
            delete_query = f"DELETE FROM `{table_ref}` WHERE TRUE"
            self.client.query(delete_query).result()

            self.logger.info(f"Created table {table_ref} with dynamic schema")
            return self.client.get_table(table_ref)

    def create_exchange_rate_table(self, table_id: str = "exchange_rates") -> bigquery.Table:
        """
        Create exchange rate table with schema

        Args:
            table_id: Table ID

        Returns:
            Table object
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        # Define schema
        schema = [
            bigquery.SchemaField("Fecha", "DATE", mode="REQUIRED", description="Exchange rate date"),
            bigquery.SchemaField("usdclp_obs", "FLOAT64", mode="NULLABLE", description="USD to CLP exchange rate"),
            bigquery.SchemaField("eurclp_obs", "FLOAT64", mode="NULLABLE", description="EUR to CLP exchange rate"),
            bigquery.SchemaField("ufclp", "FLOAT64", mode="NULLABLE", description="UF to CLP value"),
            bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP", mode="REQUIRED", description="Data ingestion timestamp"),
            bigquery.SchemaField("data_source", "STRING", mode="NULLABLE", description="Data source"),
        ]

        try:
            table = self.client.get_table(table_ref)
            self.logger.info(f"Table {table_ref} already exists")
            return table
        except NotFound:
            self.logger.info(f"Creating table {table_ref}")
            table = bigquery.Table(table_ref, schema=schema)

            # Add table description
            table.description = "Daily exchange rates (USD, EUR, UF) from CMF Chile"

            # Create table with partitioning on Fecha
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="Fecha"
            )

            # Add clustering
            table.clustering_fields = ["Fecha"]

            table = self.client.create_table(table)
            self.logger.info(f"Created table {table_ref}")
            return table

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_id: str,
        write_disposition: str = "WRITE_APPEND"
    ) -> bigquery.job.LoadJob:
        """
        Load DataFrame into BigQuery table

        Args:
            df: DataFrame to load
            table_id: Table ID
            write_disposition: Write disposition (WRITE_APPEND, WRITE_TRUNCATE, WRITE_EMPTY)

        Returns:
            Load job object
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        self.logger.info(f"Loading {len(df)} rows into {table_ref}")

        # Configure load job
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            autodetect=False,  # Use table schema
        )

        # Load data
        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )

        # Wait for job to complete
        job.result()

        self.logger.info(f"Loaded {job.output_rows} rows into {table_ref}")
        return job

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        table_id: str,
        merge_key: str = "Fecha"
    ):
        """
        Upsert (update or insert) data into BigQuery table

        Args:
            df: DataFrame to upsert
            table_id: Table ID
            merge_key: Column to use as merge key
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"
        temp_table_id = f"{table_id}_temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        temp_table_ref = f"{self.project_id}.{self.dataset_id}.{temp_table_id}"

        self.logger.info(f"Upserting {len(df)} rows into {table_ref}")

        try:
            # Load data into temporary table
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                autodetect=False,
            )

            job = self.client.load_table_from_dataframe(
                df, temp_table_ref, job_config=job_config
            )
            job.result()

            # Build MERGE query
            columns = df.columns.tolist()
            update_columns = [col for col in columns if col != merge_key]

            update_set = ", ".join([f"target.{col} = source.{col}" for col in update_columns])
            insert_columns = ", ".join(columns)
            insert_values = ", ".join([f"source.{col}" for col in columns])

            merge_query = f"""
            MERGE `{table_ref}` AS target
            USING `{temp_table_ref}` AS source
            ON target.{merge_key} = source.{merge_key}
            WHEN MATCHED THEN
                UPDATE SET {update_set}
            WHEN NOT MATCHED THEN
                INSERT ({insert_columns})
                VALUES ({insert_values})
            """

            # Execute merge
            query_job = self.client.query(merge_query)
            query_job.result()

            self.logger.info(f"Upsert completed for {table_ref}")

        finally:
            # Clean up temporary table
            self.client.delete_table(temp_table_ref, not_found_ok=True)
            self.logger.info(f"Deleted temporary table {temp_table_ref}")

    def query_table(
        self,
        table_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query BigQuery table

        Args:
            table_id: Table ID
            filters: Dictionary of column: value filters
            limit: Limit number of rows

        Returns:
            DataFrame with query results
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        query = f"SELECT * FROM `{table_ref}`"

        # Add filters
        if filters:
            where_clauses = []
            for col, value in filters.items():
                if isinstance(value, str):
                    where_clauses.append(f"{col} = '{value}'")
                else:
                    where_clauses.append(f"{col} = {value}")
            query += " WHERE " + " AND ".join(where_clauses)

        # Add limit
        if limit:
            query += f" LIMIT {limit}"

        self.logger.info(f"Executing query: {query}")

        df = self.client.query(query).to_dataframe()
        self.logger.info(f"Query returned {len(df)} rows")

        return df

    def get_latest_date(self, table_id: str, date_column: str = "Fecha") -> Optional[datetime]:
        """
        Get the latest date in the table

        Args:
            table_id: Table ID
            date_column: Date column name

        Returns:
            Latest date or None if table is empty
        """
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        query = f"""
        SELECT MAX({date_column}) as max_date
        FROM `{table_ref}`
        """

        try:
            result = self.client.query(query).to_dataframe()
            if not result.empty and result['max_date'].iloc[0] is not None:
                return pd.to_datetime(result['max_date'].iloc[0])
            return None
        except Exception as e:
            self.logger.warning(f"Error getting latest date: {str(e)}")
            return None

    def delete_date_range(
        self,
        table_id: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        date_column: str = "Fecha"
    ):
        """
        Delete records in a date range

        Args:
            table_id: Table ID
            start_date: Start date
            end_date: End date (defaults to start_date)
            date_column: Date column name
        """
        if end_date is None:
            end_date = start_date

        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        query = f"""
        DELETE FROM `{table_ref}`
        WHERE {date_column} BETWEEN '{start_date.strftime('%Y-%m-%d')}'
        AND '{end_date.strftime('%Y-%m-%d')}'
        """

        self.logger.info(f"Deleting records from {start_date} to {end_date}")

        query_job = self.client.query(query)
        query_job.result()

        self.logger.info("Delete completed")
