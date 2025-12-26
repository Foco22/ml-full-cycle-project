"""
Submit training job to Vertex AI using Python SDK.

This provides more control and flexibility than the gcloud CLI.

Usage:
    python vertex_ai/submit_training_job.py \
        --project-id your-project-id \
        --region us-central1 \
        --image-uri gcr.io/your-project/exchange-rate-training:latest
"""

import argparse
import os
from datetime import datetime
from google.cloud import aiplatform


def submit_training_job(
    project_id: str,
    region: str,
    image_uri: str,
    machine_type: str = 'n1-standard-4',
    service_account: str = None,
    enable_tensorboard: bool = False
):
    """
    Submit custom training job to Vertex AI.

    Args:
        project_id: GCP project ID
        region: GCP region
        image_uri: Container image URI
        machine_type: Machine type for training
        service_account: Service account email
        enable_tensorboard: Whether to enable Tensorboard
    """
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket='gs://ml-project-forecast'
    )

    # Generate job display name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_display_name = f'exchange-rate-training-{timestamp}'

    print("=" * 80)
    print("Submitting Vertex AI Training Job")
    print("=" * 80)
    print(f"Project ID: {project_id}")
    print(f"Region: {region}")
    print(f"Job Name: {job_display_name}")
    print(f"Image URI: {image_uri}")
    print(f"Machine Type: {machine_type}")
    print("")

    # Create custom job
    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=[
            {
                'machine_spec': {
                    'machine_type': machine_type,
                },
                'replica_count': 1,
                'container_spec': {
                    'image_uri': image_uri,
                    'args': ['--config', 'config/training_config.yaml']
                },
            }
        ]
    )

    print("Submitting job...")
    print("")

    # Submit job
    job.submit(
        service_account=service_account,
        restart_job_on_worker_restart=False,
        enable_web_access=False,
        tensorboard=None  # Can be configured if needed
    )

    print("✓ Training job submitted successfully!")
    print("")
    print("=" * 80)
    print("Job Details")
    print("=" * 80)
    print(f"Job Name: {job_display_name}")
    print(f"Job Resource Name: {job.resource_name}")
    print(f"Job State: {job.state}")
    print("")
    print("Monitor job:")
    print(f"  gcloud ai custom-jobs describe {job.name} --region={region}")
    print("")
    print("Stream logs:")
    print(f"  gcloud ai custom-jobs stream-logs {job.name} --region={region}")
    print("")
    print("Console URL:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
    print("")

    # Wait for job completion (optional)
    # Check if running in CI/CD environment (no interactive input)
    import sys
    if sys.stdin.isatty():
        # Running interactively
        wait = input("Wait for job completion? (y/n): ")
        if wait.lower() == 'y':
            print("\nWaiting for job to complete...")
            job.wait()
            print(f"\nJob completed with state: {job.state}")
    else:
        # Running in CI/CD (GitHub Actions, etc.)
        print("\nRunning in non-interactive mode (CI/CD)")
        print("Job submitted successfully. Not waiting for completion.")
        print("Monitor the job in the Vertex AI console.")

    return job


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Submit training job to Vertex AI'
    )

    parser.add_argument(
        '--project-id',
        type=str,
        default=os.getenv('GCP_PROJECT_ID'),
        required=False,
        help='GCP project ID'
    )

    parser.add_argument(
        '--region',
        type=str,
        default=os.getenv('GCP_REGION', 'us-central1'),
        help='GCP region'
    )

    parser.add_argument(
        '--image-uri',
        type=str,
        required=False,
        help='Container image URI (e.g., gcr.io/project/image:tag)'
    )

    parser.add_argument(
        '--machine-type',
        type=str,
        default='n1-standard-4',
        help='Machine type for training'
    )

    parser.add_argument(
        '--service-account',
        type=str,
        default=os.getenv('VERTEX_AI_SERVICE_ACCOUNT'),
        help='Service account email'
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.project_id:
        print("Error: --project-id is required (or set GCP_PROJECT_ID env var)")
        return 1

    # Build image URI if not provided
    if not args.image_uri:
        image_name = 'exchange-rate-training'
        image_tag = os.getenv('IMAGE_TAG', 'latest')
        args.image_uri = f'gcr.io/{args.project_id}/{image_name}:{image_tag}'
        print(f"Using default image URI: {args.image_uri}")

    # Submit job
    try:
        submit_training_job(
            project_id=args.project_id,
            region=args.region,
            image_uri=args.image_uri,
            machine_type=args.machine_type,
            service_account=args.service_account
        )
        return 0

    except Exception as e:
        print(f"\nError submitting job: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())