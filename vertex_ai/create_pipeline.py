"""
Create a Vertex AI Pipeline for scheduled model training.

This creates a pipeline definition that can be scheduled in Vertex AI.
"""

from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from kfp import compiler, dsl
import os


@dsl.pipeline(
    name="exchange-rate-training-pipeline",
    description="Scheduled training pipeline for exchange rate prediction model",
    pipeline_root="gs://ml-project-forecast/pipelines"
)
def training_pipeline(
    project_id: str = "ml-project-479423",
    region: str = "us-central1",
    image_uri: str = "us-central1-docker.pkg.dev/ml-project-479423/exchange-rate-training/exchange-rate-training:latest",
    machine_type: str = "n1-standard-4"
):
    """
    Pipeline that runs the model training on Vertex AI.

    Args:
        project_id: GCP project ID
        region: GCP region for training
        image_uri: Container image for training
        machine_type: Machine type for training
    """

    # Create custom training job component
    custom_job_task = create_custom_training_job_from_component(
        display_name="exchange-rate-model-training",
        container_uri=image_uri,
        command=["python", "-m", "src.training.train"],
        args=["--config", "config/training_config.yaml"],
        project=project_id,
        location=region,
        staging_bucket="gs://ml-project-479423-training",
        machine_type=machine_type,
        environment_variables={
            "GCP_PROJECT_ID": project_id,
            "GCS_BUCKET_DATA": "gs://ml-project-forecast/data",
            "GCS_BUCKET_MODELS": "gs://ml-project-forecast/models",
            "GCS_BUCKET_ARTIFACTS": "gs://ml-project-forecast/artifacts",
            "GCS_BUCKET_LOGS": "gs://ml-project-forecast/logs",
        }
    )


def compile_pipeline():
    """Compile the pipeline to JSON."""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="exchange_rate_training_pipeline.json"
    )
    print("✓ Pipeline compiled to: exchange_rate_training_pipeline.json")


def upload_pipeline(project_id: str = "ml-project-479423"):
    """Upload pipeline to Vertex AI."""

    # First compile
    compile_pipeline()

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location="us-central1")

    # Upload pipeline
    pipeline_job = aiplatform.PipelineJob(
        display_name="exchange-rate-training-pipeline",
        template_path="exchange_rate_training_pipeline.json",
        pipeline_root="gs://ml-project-forecast/pipelines",
        enable_caching=False,
    )

    print(f"✓ Pipeline uploaded: {pipeline_job.resource_name}")
    return pipeline_job


def create_schedule(
    project_id: str = "ml-project-479423",
    cron: str = "0 0 1 * *"  # First day of every month
):
    """
    Create a schedule for the pipeline.

    Args:
        project_id: GCP project ID
        cron: Cron expression for schedule
    """
    aiplatform.init(project=project_id, location="us-central1")

    # Create pipeline job
    pipeline_job = aiplatform.PipelineJob(
        display_name="exchange-rate-training-pipeline",
        template_path="exchange_rate_training_pipeline.json",
        pipeline_root="gs://ml-project-forecast/pipelines",
    )

    # Create schedule
    pipeline_job.create_schedule(
        cron=cron,
        display_name=f"monthly-training-schedule",
        max_concurrent_run_count=1,
        max_run_count=None,  # Run indefinitely
        service_account="ml-project-cs@ml-project-479423.iam.gserviceaccount.com"
    )

    print(f"✓ Schedule created with cron: {cron}")
    print("  View schedules at:")
    print(f"  https://console.cloud.google.com/vertex-ai/pipelines/schedules?project={project_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Vertex AI training pipeline")
    parser.add_argument("--action", choices=["compile", "upload", "schedule"],
                       default="compile", help="Action to perform")
    parser.add_argument("--cron", default="0 0 1 * *",
                       help="Cron expression (default: monthly on 1st)")

    args = parser.parse_args()

    if args.action == "compile":
        compile_pipeline()
    elif args.action == "upload":
        upload_pipeline()
    elif args.action == "schedule":
        create_schedule(cron=args.cron)