"""
Model promotion script for Resep.in.
Compares candidate model against current production and promotes if better.
Uses MLflow Model Registry for stage management.
"""

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_config import MLFLOW_EXPERIMENT_NAME, setup_mlflow

MODEL_REGISTRY_NAME = "resep-in-food-detector"


def get_production_metrics() -> dict | None:
    """Get metrics from the current Production model, if any."""
    client = MlflowClient()
    try:
        # Search for models with "Production" alias
        latest = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "production")
        run = client.get_run(latest.run_id)
        return {
            "run_id": latest.run_id,
            "version": latest.version,
            "mAP50": float(run.data.metrics.get("mAP50", 0)),
            "mAP50_95": float(run.data.metrics.get("mAP50_95", 0)),
        }
    except Exception:
        return None


def register_model(run_id: str, description: str = "") -> str:
    """
    Register a trained model from an MLflow run into the Model Registry.

    Args:
        run_id: MLflow run ID containing the model artifacts
        description: Description for this model version

    Returns:
        Model version string
    """
    client = MlflowClient()

    # Register the best.pt weight as a model
    model_uri = f"runs:/{run_id}/weights"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_REGISTRY_NAME,
    )

    if description:
        client.update_model_version(
            name=MODEL_REGISTRY_NAME,
            version=result.version,
            description=description,
        )

    print(f"Registered model version {result.version} from run {run_id}")
    return result.version


def promote_model(
    run_id: str,
    metric_name: str = "mAP50",
    min_threshold: float = 0.0,
    description: str = "",
) -> bool:
    """
    Evaluate a candidate model and promote to Production if it outperforms
    the current production model.

    Args:
        run_id: MLflow run ID of the candidate model
        metric_name: Metric to compare (default: mAP50)
        min_threshold: Minimum metric value to consider promotion
        description: Description for the model version

    Returns:
        True if model was promoted, False otherwise
    """
    setup_mlflow()
    client = MlflowClient()

    # Get candidate metrics
    candidate_run = client.get_run(run_id)
    candidate_metric = float(candidate_run.data.metrics.get(metric_name, 0))

    print(f"\nCandidate model (run {run_id[:8]}...):")
    print(f"  {metric_name}: {candidate_metric:.4f}")

    # Check minimum threshold
    if candidate_metric < min_threshold:
        print(f"  Below minimum threshold ({min_threshold}). Skipping.")
        return False

    # Register the model
    version = register_model(run_id, description=description)

    # Compare with current production
    production = get_production_metrics()
    if production:
        production_metric = production[metric_name]
        print(f"\nCurrent production model (v{production['version']}):")
        print(f"  {metric_name}: {production_metric:.4f}")

        if candidate_metric <= production_metric:
            print(f"\nCandidate does not outperform production. Registered as v{version} (staging).")
            client.set_registered_model_alias(MODEL_REGISTRY_NAME, "staging", version)
            return False

        print(f"\nCandidate outperforms production! Promoting v{version}...")
    else:
        print("\nNo existing production model. Promoting as first production model.")

    # Promote to production
    client.set_registered_model_alias(MODEL_REGISTRY_NAME, "production", version)
    print(f"Model v{version} promoted to Production!")
    return True


def list_models():
    """List all registered model versions with their metrics."""
    setup_mlflow()
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    except Exception:
        print("No registered models found.")
        return

    if not versions:
        print("No registered models found.")
        return

    print(f"\n{'Version':<10} {'Stage':<15} {'mAP50':<10} {'mAP50-95':<10} {'Run ID'}")
    print("-" * 70)

    for v in sorted(versions, key=lambda x: int(x.version)):
        try:
            run = client.get_run(v.run_id)
            mAP50 = run.data.metrics.get("mAP50", "N/A")
            mAP50_95 = run.data.metrics.get("mAP50_95", "N/A")
            if isinstance(mAP50, float):
                mAP50 = f"{mAP50:.4f}"
            if isinstance(mAP50_95, float):
                mAP50_95 = f"{mAP50_95:.4f}"
        except Exception:
            mAP50 = "N/A"
            mAP50_95 = "N/A"

        # Check aliases
        aliases = ", ".join(v.aliases) if v.aliases else "none"

        print(f"v{v.version:<9} {aliases:<15} {mAP50:<10} {mAP50_95:<10} {v.run_id[:8]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python promote_model.py list")
        print("  python promote_model.py promote <run_id> [min_threshold]")
        print("  python promote_model.py register <run_id>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        list_models()
    elif command == "promote" and len(sys.argv) >= 3:
        rid = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        promote_model(rid, min_threshold=threshold)
    elif command == "register" and len(sys.argv) >= 3:
        rid = sys.argv[2]
        register_model(rid)
    else:
        print(f"Unknown command: {command}")
