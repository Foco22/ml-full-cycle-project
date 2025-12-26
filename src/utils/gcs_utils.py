"""
Google Cloud Storage utilities
"""

from google.cloud import storage
import os
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)


def upload_to_gcs(local_path: str, gcs_path: str, blob_name: str = None, project_id: str = None):
    """
    Upload file to Google Cloud Storage

    Args:
        local_path: Path to local file
        gcs_path: GCS URI (gs://bucket/path) or bucket name
        blob_name: Destination blob name (optional if gcs_path is URI)
        project_id: GCP project ID (optional, uses default credentials)
    """
    # Parse GCS URI if provided
    if gcs_path.startswith('gs://'):
        # Extract bucket and blob from URI: gs://bucket/path/to/file
        gcs_path = gcs_path[5:]  # Remove 'gs://'
        parts = gcs_path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else blob_name
    else:
        bucket_name = gcs_path

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")


def download_from_gcs(bucket_name: str, blob_name: str, local_path: str, project_id: str):
    """
    Download file from Google Cloud Storage

    Args:
        bucket_name: GCS bucket name
        blob_name: Source blob name
        local_path: Destination local path
        project_id: GCP project ID
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")


def list_blobs(bucket_name: str, prefix: Optional[str] = None, project_id: str = None) -> List[str]:
    """
    List blobs in a GCS bucket

    Args:
        bucket_name: GCS bucket name
        prefix: Prefix filter
        project_id: GCP project ID

    Returns:
        List of blob names
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    blob_names = [blob.name for blob in blobs]

    logger.info(f"Found {len(blob_names)} blobs in gs://{bucket_name}/{prefix or ''}")
    return blob_names


def delete_blob(bucket_name: str, blob_name: str, project_id: str):
    """
    Delete a blob from GCS

    Args:
        bucket_name: GCS bucket name
        blob_name: Blob name to delete
        project_id: GCP project ID
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()
    logger.info(f"Deleted gs://{bucket_name}/{blob_name}")


def create_bucket(bucket_name: str, location: str, project_id: str):
    """
    Create a new GCS bucket

    Args:
        bucket_name: Name for the new bucket
        location: Bucket location
        project_id: GCP project ID
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    bucket.location = location

    bucket = storage_client.create_bucket(bucket)
    logger.info(f"Created bucket {bucket_name} in {location}")


# Alias for compatibility
list_gcs_files = list_blobs
