"""
Dataset manifest generator for Resep.in.
Creates a JSON manifest that fingerprints the dataset state,
enabling reproducible model-to-dataset traceability.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from version_config import (
    DATA_YAML,
    DATASET_DIR,
    DATASET_VERSION,
    MANIFEST_PATH,
    PIPELINE_STEPS,
    SOURCE_DATASETS,
)


def _count_files(directory: str, extension: str = "*") -> int:
    """Count files with given extension in a directory."""
    p = Path(directory)
    if not p.exists():
        return 0
    if extension == "*":
        return sum(1 for f in p.iterdir() if f.is_file())
    return sum(1 for f in p.glob(f"*.{extension}"))


def _hash_labels(dataset_dir: str) -> str:
    """
    Compute SHA-256 hash over all label files (sorted) in train/valid/test.
    This gives a lightweight dataset fingerprint — label changes (class edits,
    annotation fixes) affect model behavior. Image changes also change
    the file listing, so this captures both.
    """
    sha = hashlib.sha256()
    label_files = []

    for split in ["train", "valid", "test"]:
        label_dir = os.path.join(dataset_dir, split, "labels")
        if os.path.isdir(label_dir):
            for f in sorted(os.listdir(label_dir)):
                if f.endswith(".txt"):
                    label_files.append(os.path.join(label_dir, f))

    for fpath in label_files:
        with open(fpath, "rb") as f:
            sha.update(f.read())

    return sha.hexdigest()


def _get_split_counts(dataset_dir: str) -> dict:
    """Count images and labels per split."""
    counts = {}
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        counts[split] = {
            "images": _count_files(img_dir),
            "labels": _count_files(lbl_dir, "txt"),
        }
    return counts


def _read_class_list(data_yaml_path: str) -> list:
    """Read class names from data.yaml."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def generate_manifest(
    dataset_dir: str = DATASET_DIR,
    data_yaml_path: str = DATA_YAML,
    output_path: str = MANIFEST_PATH,
    version: str = DATASET_VERSION,
) -> dict:
    """
    Generate and save a dataset manifest JSON.

    Returns:
        dict: The manifest data
    """
    print(f"Generating dataset manifest for v{version}...")

    class_list = _read_class_list(data_yaml_path)
    split_counts = _get_split_counts(dataset_dir)
    dataset_hash = _hash_labels(dataset_dir)

    total_images = sum(s["images"] for s in split_counts.values())
    total_labels = sum(s["labels"] for s in split_counts.values())

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.basename(dataset_dir),
        "source_datasets": SOURCE_DATASETS,
        "class_list": class_list,
        "num_classes": len(class_list),
        "split_counts": split_counts,
        "total_images": total_images,
        "total_labels": total_labels,
        "dataset_hash": dataset_hash,
        "pipeline_steps": PIPELINE_STEPS,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  Manifest saved to: {output_path}")
    print(f"  Classes: {len(class_list)}")
    print(f"  Total images: {total_images}")
    print(f"  Dataset hash: {dataset_hash[:16]}...")

    return manifest


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    """Load an existing manifest from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found at {path}. "
            "Run generate_manifest() first or check the dataset directory."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    manifest = generate_manifest()
    print(f"\nManifest generated successfully:")
    print(json.dumps(manifest, indent=2))
