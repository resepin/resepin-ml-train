"""
Centralized version configuration for the Resep.in project.
Tracks dataset version, model version, and project constants.
"""

import os

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_normalisasi_fix")
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "raw_dataset")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")

# =============================================================================
# Dataset Version
# Bump this when: class list changes, new data added, cleaning rules change
# =============================================================================
DATASET_VERSION = "1.0.0"

# =============================================================================
# Source Datasets (from Roboflow)
# =============================================================================
SOURCE_DATASETS = [
    {
        "name": "Bahan-Makanan-Rumah-Tangga-1",
        "classes": 15,
        "source": "roboflow",
        "url": "https://universe.roboflow.com/fooddetection-5ne4k/bahan-makanan-rumah-tangga-70a2p-l29xs/dataset/1",
    },
    {
        "name": "Food-Ingredient-Detection-1",
        "classes": 92,  # approximate from yaml
        "source": "roboflow",
    },
    {
        "name": "food-ingredients-1",
        "source": "roboflow",
    },
    {
        "name": "Indonesian-Food-1",
        "source": "roboflow",
    },
    {
        "name": "Recipe-Recommendation-System-2-1",
        "source": "roboflow",
    },
]

# =============================================================================
# Final Class List (53 classes after merge + clean + normalize)
# =============================================================================
FINAL_CLASSES = [
    "beef", "bell_pepper", "cabbage", "carrot", "cauliflower", "chicken",
    "cucumber", "egg", "fish", "garlic", "ginger", "kumquat", "lemon",
    "onion", "pork", "potato", "shrimp", "chili", "tofu", "tomato",
    "apple", "fried_chicken", "meatball", "burger", "donut", "french_fries",
    "kiwi", "pineapple", "chicken_nugget", "banana", "pizza", "beef_rendang",
    "spaghetti", "steak", "fried_tofu", "fried_egg", "boiled_egg",
    "fried_tempeh", "eggplant_balado", "bitter_gourd", "bottle_gourd",
    "broccoli", "eggplant", "mackerel", "milkfish", "papaya", "pumpkin",
    "chayote", "tilapia", "shallot", "corn", "tuna", "pasta",
]

NUM_CLASSES = len(FINAL_CLASSES)

# =============================================================================
# Pipeline Steps (for manifest logging)
# =============================================================================
PIPELINE_STEPS = [
    "merge_raw_datasets",
    "class_normalization",
    "cleaning_underrepresented",
    "label_reindexing",
    "train_valid_test_split",
]
