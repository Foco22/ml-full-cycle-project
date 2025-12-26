"""
Simplified Vertex AI Pipeline for model training.

This runs the training container directly without creating nested jobs.
"""

from kfp import compiler, dsl
from google.cloud import aiplatform


@dsl.component(
    base_image='us-central1-docker.pkg.dev/ml-project-479423/exchange-rate-training/exchange-rate-training:latest'
)
def train_model(
    gcp_project_id: str,
    gcs_bucket_data: str,
    gcs_bucket_models: str,
    gcs_bucket_artifacts: str,
    gcs_bucket_logs: str
):
    """Run model training directly in the component."""
    import os
    import subprocess

    # Set environment variables
    os.environ['GCP_PROJECT_ID'] = gcp_project_id
    os.environ['GCS_BUCKET_DATA'] = gcs_bucket_data
    os.environ['GCS_BUCKET_MODELS'] = gcs_bucket_models
    os.environ['GCS_BUCKET_ARTIFACTS'] = gcs_bucket_artifacts
    os.environ['GCS_BUCKET_LOGS'] = gcs_bucket_logs

    print(f"Starting training with project: {gcp_project_id}")

    # Run training script
    result = subprocess.run(
        ['python', '-m', 'src.training.train', '--config', 'config/training_config.yaml'],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("✓ Training completed successfully!")


@dsl.pipeline(
    name="exchange-rate-training",
    description="Weekly training pipeline for exchange rate model (simplified)",
    pipeline_root="gs://ml-project-forecast/pipelines"
)
def training_pipeline(
    gcp_project_id: str = "ml-project-479423",
    gcs_bucket_data: str = "gs://ml-project-forecast/data",
    gcs_bucket_models: str = "gs://ml-project-forecast/models",
    gcs_bucket_artifacts: str = "gs://ml-project-forecast/artifacts",
    gcs_bucket_logs: str = "gs://ml-project-forecast/logs"
):
    """
    Simplified pipeline that runs training directly.

    Args:
        gcp_project_id: GCP project ID
        gcs_bucket_data: GCS bucket for data
        gcs_bucket_models: GCS bucket for models
        gcs_bucket_artifacts: GCS bucket for artifacts
        gcs_bucket_logs: GCS bucket for logs
    """

    train_task = train_model(
        gcp_project_id=gcp_project_id,
        gcs_bucket_data=gcs_bucket_data,
        gcs_bucket_models=gcs_bucket_models,
        gcs_bucket_artifacts=gcs_bucket_artifacts,
        gcs_bucket_logs=gcs_bucket_logs
    )


def compile_pipeline():
    """Compile the pipeline to JSON."""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="simple_training_pipeline.json"
    )
    print("✓ Pipeline compiled to: simple_training_pipeline.json")


def create_and_run(project_id: str = "ml-project-479423"):
    """Create and run the pipeline immediately."""

    # Compile first
    compile_pipeline()

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location="us-central1")

    # Create and run pipeline
    job = aiplatform.PipelineJob(
        display_name="exchange-rate-training",
        template_path="simple_training_pipeline.json",
        pipeline_root="gs://ml-project-forecast/pipelines",
        enable_caching=False,
        parameter_values={
            "gcp_project_id": project_id,
            "gcs_bucket_data": "gs://ml-project-forecast/data",
            "gcs_bucket_models": "gs://ml-project-forecast/models",
            "gcs_bucket_artifacts": "gs://ml-project-forecast/artifacts",
            "gcs_bucket_logs": "gs://ml-project-forecast/logs"
        }
    )

    print("Starting pipeline run...")
    job.run(sync=False)

    print(f"\n✓ Pipeline submitted!")
    print(f"Job name: {job.display_name}")
    print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project_id}")

    return job


def create_schedule(
    project_id: str = "ml-project-479423",
    cron: str = "0 0 * * 1"  # Every Monday
):
    """Create a schedule for the pipeline."""

    # Compile first
    compile_pipeline()

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location="us-central1")

    # Create pipeline job with schedule
    job = aiplatform.PipelineJob(
        display_name="exchange-rate-training-weekly",
        template_path="simple_training_pipeline.json",
        pipeline_root="gs://ml-project-forecast/pipelines",
        enable_caching=False,
        parameter_values={
            "gcp_project_id": project_id,
            "gcs_bucket_data": "gs://ml-project-forecast/data",
            "gcs_bucket_models": "gs://ml-project-forecast/models",
            "gcs_bucket_artifacts": "gs://ml-project-forecast/artifacts",
            "gcs_bucket_logs": "gs://ml-project-forecast/logs"
        }
    )

    # Create schedule
    schedule = job.create_schedule(
        cron=cron,
        display_name="weekly-training-schedule",
        max_concurrent_run_count=1,
        max_run_count=None,
        service_account="ml-project-cs@ml-project-479423.iam.gserviceaccount.com"
    )

    print(f"\n✓ Schedule created!")
    print(f"Cron: {cron} (Every Monday at midnight UTC)")
    print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines/schedules?project={project_id}")

    return schedule


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Vertex AI training pipeline")
    parser.add_argument(
        "--action",
        choices=["compile", "run", "schedule"],
        default="compile",
        help="Action to perform"
    )
    parser.add_argument(
        "--cron",
        default="0 0 * * 1",
        help="Cron expression for schedule (default: Every Monday)"
    )

    args = parser.parse_args()

    if args.action == "compile":
        compile_pipeline()
    elif args.action == "run":
        create_and_run()
    elif args.action == "schedule":
        create_schedule(cron=args.cron)