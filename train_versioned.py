"""
Versioned training script for Resep.in food ingredient detection.
Integrates YOLO training with MLflow experiment tracking and DVC dataset versioning.

Usage:
    # From command line:
    python train_versioned.py

    # From notebook:
    from train_versioned import run_training, TrainingConfig
    config = TrainingConfig(epochs=50)
    run_training(config)
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import mlflow
from ultralytics import YOLO

from dataset_manifest import generate_manifest, load_manifest
from log_results import log_full_results
from mlflow_config import get_git_commit, setup_mlflow, start_run
from version_config import (
    DATA_YAML,
    DATASET_DIR,
    DATASET_VERSION,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RUNS_DIR,
)


@dataclass
class TrainingConfig:
    """All hyperparameters for a training run."""

    # Model
    model_name: str = "yolov8n.pt"
    pretrained: bool = True

    # Dataset
    data_yaml: str = DATA_YAML

    # Training
    epochs: int = 50
    patience: int = 3
    batch: int = 16
    imgsz: int = 640
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Loss weights
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.4
    close_mosaic: int = 10

    # Other
    device: str = "0"
    workers: int = 8
    seed: int = 0
    deterministic: bool = True
    amp: bool = True
    cos_lr: bool = False
    dropout: float = 0.0

    # Output
    project: str = RUNS_DIR
    name: str = ""  # Auto-generated if empty

    # Extra training params (pass-through)
    extra_params: dict = field(default_factory=dict)

    def generate_run_name(self) -> str:
        """Generate a descriptive run name with version and timestamp."""
        model_arch = os.path.splitext(self.model_name)[0]  # e.g., "yolov8n"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{DATASET_VERSION}_{model_arch}_{timestamp}"

    def to_yolo_params(self) -> dict:
        """Convert config to YOLO model.train() keyword arguments."""
        params = {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "box": self.box,
            "cls": self.cls,
            "dfl": self.dfl,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "erasing": self.erasing,
            "close_mosaic": self.close_mosaic,
            "device": self.device,
            "workers": self.workers,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "amp": self.amp,
            "cos_lr": self.cos_lr,
            "dropout": self.dropout,
            "project": self.project,
            "name": self.name,
            "pretrained": self.pretrained,
            "verbose": True,
            "plots": True,
        }
        params.update(self.extra_params)
        return params


def _ensure_manifest(dataset_dir: str = DATASET_DIR) -> dict:
    """Load or generate the dataset manifest."""
    if os.path.exists(MANIFEST_PATH):
        manifest = load_manifest(MANIFEST_PATH)
        print(f"Loaded existing manifest: v{manifest['version']}")
    else:
        print("No manifest found. Generating...")
        manifest = generate_manifest(dataset_dir=dataset_dir)
    return manifest


def run_training(config: TrainingConfig | None = None) -> dict:
    """
    Run a versioned YOLO training with full MLflow tracking.

    Args:
        config: Training configuration. Uses defaults if None.

    Returns:
        dict with keys: 'results_dir', 'metrics', 'run_id', 'run_name'
    """
    if config is None:
        config = TrainingConfig()

    # Generate run name if not set
    if not config.name:
        config.name = config.generate_run_name()

    run_name = config.name
    model_arch = os.path.splitext(config.model_name)[0]

    print("=" * 60)
    print(f"  VERSIONED TRAINING: {run_name}")
    print("=" * 60)

    # 1. Load/generate dataset manifest
    manifest = _ensure_manifest()

    # 2. Setup MLflow
    setup_mlflow()

    # 3. Start MLflow run
    tags = {
        "dataset_version": manifest["version"],
        "dataset_hash": manifest["dataset_hash"],
        "model_arch": model_arch,
        "num_classes": str(manifest["num_classes"]),
    }

    with start_run(run_name=run_name, tags=tags) as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")

        # 4. Log all hyperparameters
        config_dict = asdict(config)
        config_dict.pop("extra_params", None)
        mlflow.log_params(config_dict)

        # Log dataset manifest as artifact
        mlflow.log_artifact(MANIFEST_PATH, artifact_path="dataset")

        # 5. Train YOLO model
        print(f"\nLoading model: {config.model_name}")
        model = YOLO(config.model_name)

        print("Starting training...")
        yolo_params = config.to_yolo_params()
        result = model.train(**yolo_params)

        # YOLO's built-in MLflow callback ends the run after training.
        # Re-activate it so we can continue logging custom metrics.
        if mlflow.active_run() is None:
            mlflow.start_run(run_id=run_id)

        # 6. Determine results directory
        results_dir = os.path.join(config.project, run_name)
        if not os.path.exists(results_dir):
            # YOLO might append a number if name already exists
            results_dir = str(result.save_dir) if hasattr(result, "save_dir") else results_dir

        print(f"\nResults saved to: {results_dir}")

        # 7. Log results to MLflow
        final_metrics = log_full_results(results_dir)

        # 8. Save run metadata alongside YOLO output
        run_metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "model_arch": model_arch,
            "dataset_version": manifest["version"],
            "dataset_hash": manifest["dataset_hash"],
            "git_commit": get_git_commit(),
            "config": config_dict,
            "final_metrics": final_metrics,
        }
        metadata_path = os.path.join(results_dir, "run_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)
        mlflow.log_artifact(metadata_path, artifact_path="metadata")

        print("\n" + "=" * 60)
        print(f"  TRAINING COMPLETE: {run_name}")
        print(f"  mAP50: {final_metrics.get('mAP50', 'N/A')}")
        print(f"  mAP50-95: {final_metrics.get('mAP50_95', 'N/A')}")
        print(f"  MLflow Run ID: {run_id}")
        print(f"  Results: {results_dir}")
        print("=" * 60)

    return {
        "results_dir": results_dir,
        "metrics": final_metrics,
        "run_id": run_id,
        "run_name": run_name,
    }


if __name__ == "__main__":
    # Default training with YOLOv8n
    config = TrainingConfig(
        model_name="yolov8n.pt",
        epochs=50,
        patience=3,
        batch=16,
        imgsz=640,
    )
    result = run_training(config)
