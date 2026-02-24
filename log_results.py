"""
Parse YOLO training results and log them to MLflow.
Handles results.csv parsing, artifact logging, and metric extraction.
"""

import os
from pathlib import Path

import mlflow
import pandas as pd


def parse_results_csv(results_dir: str) -> pd.DataFrame:
    """
    Parse YOLO results.csv from a training run directory.

    Args:
        results_dir: Path to the YOLO training output directory

    Returns:
        DataFrame with training metrics per epoch
    """
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # YOLO sometimes adds spaces in column names
    df.columns = df.columns.str.strip()
    return df


def get_final_metrics(results_dir: str) -> dict:
    """
    Extract final epoch metrics from a YOLO training run.

    Returns:
        dict with metric names and their final values
    """
    df = parse_results_csv(results_dir)
    last_row = df.iloc[-1]

    # Map YOLO column names to clean metric names
    metric_mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
        "lr/pg0": "lr_pg0",
        "lr/pg1": "lr_pg1",
        "lr/pg2": "lr_pg2",
    }

    metrics = {}
    for yolo_name, clean_name in metric_mapping.items():
        if yolo_name in last_row.index:
            metrics[clean_name] = float(last_row[yolo_name])

    return metrics


def log_epoch_metrics(results_dir: str):
    """Log all epoch-by-epoch metrics to MLflow for charting."""
    df = parse_results_csv(results_dir)

    metric_cols = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
    }

    for _, row in df.iterrows():
        epoch = int(row.get("epoch", row.name))
        step_metrics = {}
        for yolo_name, clean_name in metric_cols.items():
            if yolo_name in row.index:
                step_metrics[clean_name] = float(row[yolo_name])
        mlflow.log_metrics(step_metrics, step=epoch)

    print(f"  Logged {len(df)} epochs of metrics to MLflow")


def log_training_artifacts(results_dir: str):
    """
    Log all important training artifacts to MLflow.
    Includes weights, plots, configs, and results.
    """
    results_path = Path(results_dir)

    # Log weight files
    weights_dir = results_path / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            mlflow.log_artifact(str(weight_file), artifact_path="weights")
            print(f"  Logged weight: {weight_file.name}")

    # Log configuration
    args_yaml = results_path / "args.yaml"
    if args_yaml.exists():
        mlflow.log_artifact(str(args_yaml), artifact_path="config")

    # Log results CSV
    results_csv = results_path / "results.csv"
    if results_csv.exists():
        mlflow.log_artifact(str(results_csv), artifact_path="results")

    # Log plots
    plot_extensions = ["*.png", "*.jpg"]
    plots_logged = 0
    for ext in plot_extensions:
        for plot_file in results_path.glob(ext):
            mlflow.log_artifact(str(plot_file), artifact_path="plots")
            plots_logged += 1

    print(f"  Logged {plots_logged} plot files to MLflow")


def log_full_results(results_dir: str):
    """
    Convenience function: log final metrics, epoch metrics, and artifacts.

    Args:
        results_dir: Path to YOLO training output directory

    Returns:
        dict: Final metrics
    """
    print(f"\nLogging results from: {results_dir}")

    # Log final metrics as summary
    final_metrics = get_final_metrics(results_dir)
    mlflow.log_metrics(final_metrics)
    print(f"  Final mAP50: {final_metrics.get('mAP50', 'N/A'):.4f}")
    print(f"  Final mAP50-95: {final_metrics.get('mAP50_95', 'N/A'):.4f}")

    # Log epoch-by-epoch metrics
    log_epoch_metrics(results_dir)

    # Log artifacts
    log_training_artifacts(results_dir)

    return final_metrics
