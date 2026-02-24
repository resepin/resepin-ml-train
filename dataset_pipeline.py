"""
Dataset pipeline for Resep.in.
Converts notebook cells from dataset.ipynb into importable, reproducible functions.
Each function corresponds to a pipeline step tracked in the manifest.

Usage:
    from dataset_pipeline import run_full_pipeline
    run_full_pipeline()
"""

import os
import shutil
from collections import OrderedDict
from pathlib import Path

import yaml

from version_config import (
    DATASET_DIR,
    DATASET_VERSION,
    FINAL_CLASSES,
    RAW_DATASET_DIR,
)

# Class normalization map: maps various raw class names to final standardized names
CLASS_MAP = {
    # Indonesian -> English
    "ayam": "chicken",
    "sapi": "beef",
    "bawangm": "onion",
    "bawangp": "garlic",
    "brokoli": "broccoli",
    "cabai": "chili",
    "jagung": "corn",
    "jahe": "ginger",
    "kol": "cabbage",
    "telur": "egg",
    "terong": "eggplant",
    "timun": "cucumber",
    "tomat": "tomato",
    "tongkol": "tuna",
    "wortel": "carrot",
    # Indonesian-Food dataset
    "apel": "apple",
    "ayam goreng": "fried_chicken",
    "bakso": "meatball",
    "donat": "donut",
    "kentang goreng": "french_fries",
    "nanas": "pineapple",
    "nugget": "chicken_nugget",
    "pisang": "banana",
    "rendang sapi": "beef_rendang",
    "tahu goreng": "fried_tofu",
    "telur goreng": "fried_egg",
    "telur rebus": "boiled_egg",
    "tempe goreng": "fried_tempeh",
    "terong balado": "eggplant_balado",
    # English variations
    "bell pepper": "bell_pepper",
    "chicken breast": "chicken",
    "chicken wing": "chicken",
    "chilli": "chili",
    "bean sprout": "bean_sprout",
    "bok choy": "bok_choy",
    "bitter gourd": "bitter_gourd",
    "bottle gourd": "bottle_gourd",
}


def merge_datasets(
    raw_dir: str = RAW_DATASET_DIR,
    output_dir: str = None,
) -> str:
    """
    Merge multiple raw Roboflow datasets into one, re-indexing class IDs.

    Args:
        raw_dir: Directory containing raw dataset subdirectories
        output_dir: Where to write the merged dataset

    Returns:
        Path to the merged dataset directory
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(raw_dir), "merged_dataset")

    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Read all dataset class lists
    datasets = {}
    for ds_name in sorted(os.listdir(raw_dir)):
        ds_path = os.path.join(raw_dir, ds_name)
        yaml_path = os.path.join(ds_path, "data.yaml")
        if os.path.isdir(ds_path) and os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            datasets[ds_name] = {
                "path": ds_path,
                "classes": data.get("names", []),
            }

    print(f"Found {len(datasets)} datasets to merge:")
    for name, info in datasets.items():
        print(f"  {name}: {len(info['classes'])} classes")

    # Build unified class list
    all_classes = []
    for info in datasets.values():
        for cls in info["classes"]:
            normalized = CLASS_MAP.get(cls.lower(), cls.lower().replace(" ", "_"))
            if normalized not in all_classes:
                all_classes.append(normalized)

    merged_classes = list(OrderedDict.fromkeys(all_classes))
    print(f"\nTotal merged classes: {len(merged_classes)}")

    # Build class ID mapping per dataset
    class_mappings = {}
    for name, info in datasets.items():
        mapping = {}
        for old_id, cls in enumerate(info["classes"]):
            normalized = CLASS_MAP.get(cls.lower(), cls.lower().replace(" ", "_"))
            new_id = merged_classes.index(normalized)
            mapping[old_id] = new_id
        class_mappings[name] = mapping

    # Copy files and remap labels
    total_copied = 0
    for ds_name, info in datasets.items():
        ds_path = info["path"]
        id_map = class_mappings[ds_name]

        for split in ["train", "valid", "test"]:
            img_src = os.path.join(ds_path, split, "images")
            lbl_src = os.path.join(ds_path, split, "labels")

            if not os.path.isdir(img_src):
                continue

            img_dst = os.path.join(output_dir, split, "images")
            lbl_dst = os.path.join(output_dir, split, "labels")

            for img_file in os.listdir(img_src):
                src_img = os.path.join(img_src, img_file)
                dst_img = os.path.join(img_dst, img_file)
                if os.path.isfile(src_img):
                    shutil.copy2(src_img, dst_img)

                    # Remap label file
                    lbl_file = os.path.splitext(img_file)[0] + ".txt"
                    src_lbl = os.path.join(lbl_src, lbl_file)
                    dst_lbl = os.path.join(lbl_dst, lbl_file)

                    if os.path.isfile(src_lbl):
                        with open(src_lbl, "r") as f:
                            lines = f.readlines()

                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                old_id = int(parts[0])
                                new_id = id_map.get(old_id, old_id)
                                parts[0] = str(new_id)
                                new_lines.append(" ".join(parts))

                        with open(dst_lbl, "w") as f:
                            f.write("\n".join(new_lines) + "\n")

                    total_copied += 1

    # Write data.yaml
    yaml_data = {
        "path": output_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": merged_classes,
    }
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"\nMerge complete. {total_copied} images copied to {output_dir}")
    return output_dir


def clean_dataset(
    input_dir: str,
    output_dir: str = None,
    min_samples: int = 10,
    classes_to_keep: list = None,
) -> str:
    """
    Remove underrepresented classes and clean the dataset.

    Args:
        input_dir: Merged dataset directory
        output_dir: Where to write cleaned dataset
        min_samples: Minimum samples per class to keep
        classes_to_keep: Explicit list of classes to keep (overrides min_samples)

    Returns:
        Path to cleaned dataset
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), "cleaned_dataset")

    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Read current class list
    with open(os.path.join(input_dir, "data.yaml"), "r") as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", [])

    # Count class occurrences in training labels
    class_counts = {i: 0 for i in range(len(class_names))}
    train_labels = os.path.join(input_dir, "train", "labels")
    if os.path.isdir(train_labels):
        for lf in os.listdir(train_labels):
            if lf.endswith(".txt"):
                with open(os.path.join(train_labels, lf), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            class_counts[cid] = class_counts.get(cid, 0) + 1

    # Determine which class IDs to keep
    if classes_to_keep:
        keep_ids = {i for i, name in enumerate(class_names) if name in classes_to_keep}
    else:
        keep_ids = {cid for cid, count in class_counts.items() if count >= min_samples}

    removed = [class_names[i] for i in range(len(class_names)) if i not in keep_ids]
    kept = [class_names[i] for i in sorted(keep_ids)]

    print(f"Classes kept: {len(kept)}")
    print(f"Classes removed: {len(removed)}")
    if removed:
        print(f"  Removed: {', '.join(removed[:10])}{'...' if len(removed) > 10 else ''}")

    # Build old-to-new ID mapping
    new_class_list = []
    old_to_new = {}
    for old_id in sorted(keep_ids):
        new_id = len(new_class_list)
        old_to_new[old_id] = new_id
        new_class_list.append(class_names[old_id])

    # Copy and remap
    total = 0
    for split in ["train", "valid", "test"]:
        img_src = os.path.join(input_dir, split, "images")
        lbl_src = os.path.join(input_dir, split, "labels")
        if not os.path.isdir(img_src):
            continue

        for img_file in os.listdir(img_src):
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            src_lbl = os.path.join(lbl_src, lbl_file)

            if not os.path.isfile(src_lbl):
                continue

            # Read and filter labels
            with open(src_lbl, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    old_id = int(parts[0])
                    if old_id in old_to_new:
                        parts[0] = str(old_to_new[old_id])
                        new_lines.append(" ".join(parts))

            # Only copy if there are remaining annotations
            if new_lines:
                shutil.copy2(
                    os.path.join(img_src, img_file),
                    os.path.join(output_dir, split, "images", img_file),
                )
                with open(os.path.join(output_dir, split, "labels", lbl_file), "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                total += 1

    # Write data.yaml
    yaml_data = {
        "path": output_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": new_class_list,
    }
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"\nCleaning complete. {total} images kept in {output_dir}")
    return output_dir


def generate_data_yaml(
    dataset_dir: str = DATASET_DIR,
    class_list: list = None,
) -> str:
    """
    Generate a data.yaml for the dataset.

    Args:
        dataset_dir: Dataset directory
        class_list: Class names list (defaults to FINAL_CLASSES)

    Returns:
        Path to generated data.yaml
    """
    if class_list is None:
        class_list = FINAL_CLASSES

    yaml_data = {
        "path": dataset_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": class_list,
    }

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"data.yaml written to {yaml_path} with {len(class_list)} classes")
    return yaml_path


def run_full_pipeline(
    raw_dir: str = RAW_DATASET_DIR,
    final_dir: str = DATASET_DIR,
    min_samples: int = 10,
    version: str = DATASET_VERSION,
) -> str:
    """
    Run the full dataset pipeline: merge → clean → finalize → manifest.

    Args:
        raw_dir: Directory with raw Roboflow datasets
        final_dir: Output directory for the final dataset
        min_samples: Minimum samples per class for cleaning
        version: Dataset version tag

    Returns:
        Path to the final dataset directory
    """
    from dataset_manifest import generate_manifest

    print("=" * 60)
    print(f"  DATASET PIPELINE v{version}")
    print("=" * 60)

    # Step 1: Merge
    print("\n[Step 1/4] Merging raw datasets...")
    merged_dir = merge_datasets(raw_dir=raw_dir)

    # Step 2: Clean
    print("\n[Step 2/4] Cleaning dataset...")
    cleaned_dir = clean_dataset(input_dir=merged_dir, min_samples=min_samples)

    # Step 3: Copy to final location
    print("\n[Step 3/4] Finalizing dataset...")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    shutil.copytree(cleaned_dir, final_dir)
    print(f"  Final dataset at: {final_dir}")

    # Step 4: Generate manifest
    print("\n[Step 4/4] Generating manifest...")
    generate_manifest(dataset_dir=final_dir, version=version)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE")
    print("=" * 60)

    return final_dir


if __name__ == "__main__":
    run_full_pipeline()
