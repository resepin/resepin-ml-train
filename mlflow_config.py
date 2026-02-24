"""
MLflow configuration and helpers for Resep.in experiment tracking.
"""

import os
import subprocess

import mlflow

from version_config import PROJECT_ROOT

# =============================================================================
# MLflow Settings
# =============================================================================
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db".replace("\\", "/")
MLFLOW_EXPERIMENT_NAME = "resep-in-food-detection"


def setup_mlflow() -> str:
    """
    Configure MLflow tracking. Call this before starting any experiment.

    Returns:
        str: The experiment ID
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {experiment.name} (ID: {experiment.experiment_id})")
    return experiment.experiment_id


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def start_run(
    run_name: str,
    tags: dict | None = None,
    description: str = "",
) -> mlflow.ActiveRun:
    """
    Start an MLflow run with standard tags.

    Args:
        run_name: Human-readable name for this run
        tags: Additional tags to set
        description: Run description

    Returns:
        mlflow.ActiveRun context manager
    """
    default_tags = {
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
    }
    if tags:
        default_tags.update(tags)

    return mlflow.start_run(
        run_name=run_name,
        tags=default_tags,
        description=description,
    )
