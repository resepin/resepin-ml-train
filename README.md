# Resep.in — Food Ingredient Detection with Full ML Versioning

A YOLOv8-based object detection system that identifies **53 food ingredients** from images, built for the Resep.in recipe recommendation platform. This repository includes a complete **ML versioning pipeline** using DVC (dataset versioning), MLflow (experiment tracking), and Google Drive (remote storage).

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Dataset Setup (DVC Pull)](#dataset-setup-dvc-pull)
- [Training a Model](#training-a-model)
- [Viewing Experiment Results (MLflow)](#viewing-experiment-results-mlflow)
- [Model Promotion](#model-promotion)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Detected Classes (53)](#detected-classes-53)
- [How DVC + MLflow + Google Drive Work Together](#how-dvc--mlflow--google-drive-work-together)
- [Versioning Workflow Summary](#versioning-workflow-summary)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Overview

**Resep.in** detects food ingredients in photos so users can get recipe suggestions based on what they have. The ML side of the project works as follows:

1. **5 Roboflow datasets** are merged and normalized into a single clean dataset of **~50,800 images** across **53 ingredient classes**.
2. **YOLOv8 Nano** is trained on this dataset with full hyperparameter tracking.
3. Every training run is **versioned end-to-end**: the exact dataset state, code commit, hyperparameters, and resulting metrics are all recorded.
4. The best model is promoted to **"production"** via a model registry.

| Item | Detail |
|------|--------|
| Model | YOLOv8n (Ultralytics) |
| Classes | 53 food ingredients |
| Dataset Size | ~50,800 images (~30 GB) |
| Framework | PyTorch + Ultralytics |
| Experiment Tracking | MLflow (local SQLite) |
| Dataset Versioning | DVC → Google Drive |
| Code Versioning | Git → GitHub |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      YOUR MACHINE                            │
│                                                              │
│   Code (small files)              Data (large files ~30 GB)  │
│   ┌──────────────────┐           ┌────────────────────────┐  │
│   │ *.py, *.ipynb    │           │ dataset_normalisasi_fix │  │
│   │ *.yaml, *.txt    │           │ raw_dataset/            │  │
│   │ requirements.txt │           │ yolov8n.pt              │  │
│   └───────┬──────────┘           └──────────┬─────────────┘  │
│           │                                  │                │
│      Git tracks                        DVC tracks             │
│      (full files)                (.dvc pointer files only)    │
│           │                                  │                │
│           │               ┌──────────────────┘                │
│           │               │                                   │
│           │        ┌──────▼──────────────────────┐            │
│           │        │        MLflow (mlflow.db)    │            │
│           │        │  Every training run logs:    │            │
│           │        │  • All hyperparameters       │            │
│           │        │  • Epoch-by-epoch metrics    │            │
│           │        │  • Model weights (best.pt)   │            │
│           │        │  • Dataset hash & version    │            │
│           │        │  • Git commit hash           │            │
│           │        └──────────────────────────────┘            │
│           │                                  │                │
└───────────┼──────────────────────────────────┼────────────────┘
            │                                  │
       git push                           dvc push
            │                                  │
   ┌────────▼─────────┐            ┌───────────▼──────────────┐
   │     GitHub        │            │     Google Drive          │
   │  Code + .dvc      │            │  Actual images, labels,  │
   │  pointer files    │            │  weights (~30 GB)         │
   └──────────────────┘            └──────────────────────────┘
```

**Key insight:** Git stores code + tiny `.dvc` pointer files. DVC stores the actual large data on Google Drive. MLflow connects them by recording which dataset hash + code commit + hyperparameters produced which model.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8+ (tested on 3.12) | |
| Git | 2.30+ | For version control |
| NVIDIA GPU | CUDA-capable | Required for training |
| CUDA Toolkit | 11.8 or 12.x | Must match PyTorch build |
| pip | Latest | `python -m pip install --upgrade pip` |
| Google Account | — | For DVC remote (Google Drive) |

> **RAM requirement:** At least 16 GB RAM recommended. The dataset is ~30 GB on disk.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/resepin/resepin-ml-train.git
cd resepin-ml-train
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Install all Python packages
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust cu118 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DVC Google Drive support
pip install dvc-gdrive
```

### 4. Verify GPU is available

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True, Device: NVIDIA GeForce RTX XXXX
```

---

## Dataset Setup (DVC Pull)

The dataset (~30 GB) is **not stored in Git**. Instead, Git contains tiny `.dvc` pointer files that tell DVC where to download the data from Google Drive.

### Download the dataset

```bash
# This downloads all DVC-tracked files from Google Drive
dvc pull
```

On first run, DVC will open a browser for Google OAuth authentication. Sign in with a Google account that has access to the shared Drive folder.

After `dvc pull` completes, you should see:

```
resep.in/
├── dataset_normalisasi_fix/     ← ~50,800 images + labels (downloaded)
│   ├── train/images/            ← 40,696 training images
│   ├── valid/images/            ← 5,088 validation images
│   ├── test/images/             ← 5,088 test images
│   └── data.yaml                ← YOLO dataset config
├── raw_dataset/                 ← 5 original Roboflow datasets (downloaded)
└── yolov8n.pt                   ← Pre-trained YOLOv8 Nano weights (downloaded)
```

### Verify the dataset

```bash
# Check that all DVC files are present
dvc status
```

If everything is up to date, the output will be empty or show `Data and calculation files are up to date.`

---

## Training a Model

There are two ways to train: the **Jupyter notebook** (recommended for interactive use) or the **Python script** (recommended for reproducibility).

### Option A: Jupyter Notebook (Recommended)

Open `train_model.ipynb` and run all cells. The notebook:

1. Checks GPU availability
2. Shows dataset statistics
3. Trains YOLOv8n with MLflow tracking enabled
4. Logs all metrics, hyperparameters, and artifacts automatically

The training cell uses the versioned pipeline:

```python
from train_versioned import run_training, TrainingConfig

config = TrainingConfig(
    model_name="yolov8n.pt",
    epochs=50,          # Total epochs
    patience=3,         # Early stopping patience
    batch=16,           # Batch size (reduce if GPU OOM)
    imgsz=640,          # Image size
    optimizer="AdamW",  # Optimizer
    lr0=0.001,          # Initial learning rate
    device="0",         # GPU device (0 = first GPU)
)

result = run_training(config)
```

### Option B: Command Line

```bash
python train_versioned.py
```

This runs training with default settings (50 epochs, batch=16, AdamW optimizer).

### Customizing Hyperparameters

All hyperparameters are in the `TrainingConfig` dataclass. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"yolov8n.pt"` | Base model (yolov8n/s/m/l/x) |
| `epochs` | `50` | Maximum training epochs |
| `patience` | `3` | Early stopping patience |
| `batch` | `16` | Batch size (reduce for less VRAM) |
| `imgsz` | `640` | Input image resolution |
| `optimizer` | `"AdamW"` | Optimizer (SGD, Adam, AdamW) |
| `lr0` | `0.001` | Initial learning rate |
| `device` | `"0"` | GPU index (`"cpu"` for CPU) |

### What happens during training

When you call `run_training(config)`, the following happens automatically:

1. **Dataset manifest** is loaded/generated — fingerprints the exact dataset state (SHA-256 hash of all labels)
2. **MLflow run** starts — creates a tracked experiment run
3. **All hyperparameters** are logged to MLflow
4. **YOLO trains** the model using Ultralytics
5. **Epoch-by-epoch metrics** (loss, mAP, precision, recall) are logged to MLflow
6. **Model weights** (`best.pt`, `last.pt`) are saved as MLflow artifacts
7. **Training plots** (confusion matrix, PR curve, loss curves) are saved as artifacts
8. **Run metadata** JSON is saved linking everything together

### Training output

Results are saved to `runs/detect/<run_name>/`:

```
runs/detect/v1.0.0_yolov8n_20260224_143000/
├── weights/
│   ├── best.pt              ← Best model weights (use this for inference)
│   └── last.pt              ← Last epoch weights
├── results.csv              ← Epoch-by-epoch metrics
├── confusion_matrix.png     ← Confusion matrix plot
├── PR_curve.png             ← Precision-Recall curve
├── F1_curve.png             ← F1 score curve
├── results.png              ← Training curves plot
├── args.yaml                ← Full YOLO config used
└── run_metadata.json        ← MLflow run ID + dataset hash link
```

---

## Viewing Experiment Results (MLflow)

MLflow provides a web dashboard to compare all your training runs.

### Launch the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### What you can see

- **Runs table** — Every training run with all hyperparameters and final metrics side-by-side
- **Metric charts** — Epoch-by-epoch plots of loss, mAP, precision, recall
- **Artifact browser** — Download model weights, training plots, and configs from any run
- **Run comparison** — Select multiple runs and compare their metrics visually
- **Tags** — Each run is tagged with `dataset_version`, `dataset_hash`, `model_arch`, `git_commit`

### Key metrics tracked

| Metric | Description |
|--------|-------------|
| `mAP50` | Mean Average Precision at IoU=0.50 (primary metric) |
| `mAP50_95` | Mean Average Precision at IoU=0.50:0.95 |
| `precision` | Detection precision |
| `recall` | Detection recall |
| `train_box_loss` | Training bounding box loss |
| `train_cls_loss` | Training classification loss |
| `val_box_loss` | Validation bounding box loss |
| `val_cls_loss` | Validation classification loss |

---

## Model Promotion

After training, you can register and promote the best model using the Model Registry.

### From the notebook

The `train_model.ipynb` notebook has cells for model promotion:

```python
from promote_model import promote_model, list_models

# Promote the best model to production (compares against current production)
promoted = promote_model(
    run_id=result["run_id"],     # From training result
    metric_name="mAP50",         # Compare on mAP50
    min_threshold=0.3,           # Minimum mAP50 to consider
)

# List all registered model versions
list_models()
```

### From the command line

```bash
# List all registered models
python promote_model.py list

# Promote a model by its MLflow run ID
python promote_model.py promote <run_id> 0.3

# Just register without promoting
python promote_model.py register <run_id>
```

### How promotion works

1. The candidate model's `mAP50` is compared against the current `production` model
2. If the candidate is better (higher mAP50), it becomes the new `production` model
3. If not, it is registered as `staging` for reference
4. If no production model exists yet, the first registered model becomes production

---

## Project Structure

```
resep.in/
│
├── train_model.ipynb              # Main training notebook (start here)
├── dataset.ipynb                  # Dataset preprocessing/merging notebook
├── train_versioned.py             # Versioned training with MLflow integration
├── version_config.py              # Centralized project config & constants
├── dataset_manifest.py            # Dataset fingerprinting (SHA-256 hash)
├── mlflow_config.py               # MLflow setup & helpers
├── log_results.py                 # YOLO results parser → MLflow logger
├── promote_model.py               # Model registry & promotion logic
├── dataset_pipeline.py            # Dataset merge/clean/normalize pipeline
├── requirements.txt               # Python dependencies
├── yolov8n.pt                     # Pre-trained YOLOv8 Nano (DVC-tracked)
├── yolov8n.pt.dvc                 # DVC pointer for yolov8n.pt
│
├── dataset_normalisasi_fix/       # Final clean dataset (DVC-tracked, ~30 GB)
│   ├── data.yaml                  # YOLO dataset configuration (53 classes)
│   ├── manifest.json              # Dataset fingerprint (version + hash)
│   ├── train/images/              # 40,696 training images
│   ├── train/labels/              # Training labels (YOLO format)
│   ├── valid/images/              # 5,088 validation images
│   ├── valid/labels/              # Validation labels
│   ├── test/images/               # 5,088 test images
│   └── test/labels/               # Test labels
├── dataset_normalisasi_fix.dvc    # DVC pointer for the dataset
│
├── raw_dataset/                   # 5 original Roboflow source datasets
│   ├── Bahan-Makanan-Rumah-Tangga-1/
│   ├── Food-Ingredient-Detection-1/
│   ├── food-ingredients-1/
│   ├── Indonesian-Food-1/
│   └── Recipe-Recommendation-System-2-1/
├── raw_dataset.dvc                # DVC pointer for raw datasets
│
├── runs/detect/                   # YOLO training outputs
│   └── training_results/          # Previous training run
│
├── mlflow.db                      # MLflow experiment database (SQLite)
├── .dvc/                          # DVC configuration
│   └── config                     # DVC remote settings (Google Drive)
├── .gitignore                     # Git ignore rules
│
├── train_model_medium/            # YOLOv8m training experiments
│   └── train_model_medium.ipynb
└── unuse/                         # Archived/unused experiments
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `version_config.py` | Central configuration: project paths, dataset version (`1.0.0`), 53 class names, source dataset list, pipeline step names |
| `dataset_manifest.py` | Generates `manifest.json` with SHA-256 hash of all labels, split counts, class list. Ensures dataset traceability |
| `mlflow_config.py` | Sets up MLflow with SQLite backend (`mlflow.db`), provides `setup_mlflow()`, `start_run()`, git hash helpers |
| `log_results.py` | Parses YOLO `results.csv`, logs epoch-by-epoch metrics to MLflow, uploads weights/plots/configs as artifacts |
| `train_versioned.py` | `TrainingConfig` dataclass (all hyperparameters) + `run_training()` function that wraps YOLO training with full MLflow integration |
| `dataset_pipeline.py` | Merges 5 Roboflow datasets, normalizes class names (Indonesian → English), cleans underrepresented classes, generates `data.yaml` |
| `promote_model.py` | Model Registry: `register_model()`, `promote_model()` (compares candidate vs production on mAP50), `list_models()` |

---

## Detected Classes (53)

The model detects 53 food ingredients organized into the following categories:

| Category | Classes |
|----------|---------|
| **Meat & Protein** | beef, chicken, pork, fish, shrimp, tuna, mackerel, milkfish, tilapia, egg |
| **Vegetables** | cabbage, carrot, cauliflower, cucumber, bell_pepper, chili, broccoli, eggplant, corn, potato, pumpkin, chayote, bitter_gourd, bottle_gourd, tomato |
| **Aromatics** | garlic, ginger, onion, shallot |
| **Soy Products** | tofu, fried_tofu, fried_tempeh |
| **Fruits** | apple, banana, kiwi, kumquat, lemon, papaya, pineapple |
| **Prepared Foods** | fried_chicken, fried_egg, boiled_egg, meatball, burger, pizza, spaghetti, pasta, steak, french_fries, donut, chicken_nugget |
| **Indonesian Dishes** | beef_rendang, eggplant_balado |

---

## How DVC + MLflow + Google Drive Work Together

### The Problem

This project has **two types of files**:
1. **Code** (small, text-based): Python scripts, notebooks, YAML configs — ~KB each
2. **Data** (large, binary): 50,800 images, labels, model weights — ~30 GB total

Git handles code well but cannot store 30 GB of images efficiently. The solution uses three tools, each handling a specific layer:

### Layer 1: Git + GitHub → Code Versioning

- Git tracks **all source code**: `.py`, `.ipynb`, `.yaml`, `.txt`, `requirements.txt`
- Git also tracks **`.dvc` pointer files** — these are tiny (~200 bytes) files that contain a hash pointing to the actual data
- GitHub hosts the remote repository

### Layer 2: DVC + Google Drive → Data Versioning

- **DVC (Data Version Control)** works alongside Git specifically for large files
- When you run `dvc add dataset_normalisasi_fix/`, DVC:
  1. Computes a hash of every file in the folder
  2. Creates `dataset_normalisasi_fix.dvc` (a tiny pointer file with the hash)
  3. Adds the actual folder to `.gitignore` (Git won't track it)
- `dvc push` uploads the actual data to **Google Drive** (our remote storage)
- `dvc pull` downloads the data matching the hash in the `.dvc` pointer file
- **Result:** Git only stores the pointer (tiny), Google Drive stores the data (large)

### Layer 3: MLflow → Experiment Tracking

- MLflow records every training run in a local **SQLite database** (`mlflow.db`)
- Each run logs:
  - All hyperparameters (learning rate, batch size, epochs, augmentations, etc.)
  - Epoch-by-epoch metrics (loss, mAP50, mAP50-95, precision, recall)
  - Artifacts (best.pt weights, confusion matrix, PR curve, loss plots)
  - Tags: `dataset_version`, `dataset_hash`, `git_commit`, `model_arch`
- The **Model Registry** lets you tag model versions as `production` or `staging`

### How They Connect

```
Git Commit ──────────┐
(code version)       │
                     ├──→ MLflow Run ──→ Metrics + Model
Dataset Hash ────────┘       │
(from DVC manifest)          │
                             ▼
                     Model Registry
                     (production / staging)
```

Every MLflow run records the **git commit** (which code was used) and the **dataset hash** (which data was used). This means you can always trace back:

> "Model v3 was trained with commit `a1b2c3d` on dataset v1.0.0 (hash `848bb219...`) with batch=16, lr=0.001, and achieved mAP50=0.72"

### Reproducibility Chain

When someone wants to reproduce your exact results:

```bash
git checkout <commit_hash>   # Get the exact code version
dvc checkout                 # Get the exact dataset version matching that commit
python train_versioned.py    # Train with the same hyperparameters (logged in MLflow)
```

---

## Versioning Workflow Summary

### When you change the dataset

```bash
# 1. Make your changes to dataset_normalisasi_fix/
# 2. Update version in version_config.py
# 3. Re-track with DVC
dvc add dataset_normalisasi_fix

# 4. Regenerate manifest
python -c "from dataset_manifest import generate_manifest; generate_manifest()"

# 5. Commit the updated pointer + manifest
git add dataset_normalisasi_fix.dvc dataset_normalisasi_fix/manifest.json version_config.py
git commit -m "dataset: bump to v1.1.0 - added new images"

# 6. Push data and code
dvc push        # Upload new data to Google Drive
git push        # Push code + pointer to GitHub
```

### When you retrain

```bash
# 1. Train (MLflow logs everything automatically)
python train_versioned.py

# 2. Check results in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 3. Promote the best model
python promote_model.py promote <run_id> 0.3

# 4. Commit any new outputs
git add .
git commit -m "training: v1.0.0 yolov8n mAP50=0.72"
git push
```

### When a collaborator joins

```bash
git clone https://github.com/resepin/resepin-ml-train.git
cd resepin-ml-train
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dvc-gdrive
dvc pull                  # Downloads ~30 GB dataset from Google Drive
# Open train_model.ipynb and run all cells
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `dvc pull` asks for authentication | Sign in with a Google account that has access to the shared Drive folder |
| `CUDA out of memory` | Reduce `batch` size (try 8 or 4) in `TrainingConfig` |
| `dvc push` SSL errors | Retry — DVC resumes automatically. Run `dvc push -j 1` to reduce parallel jobs |
| `mlflow ui` won't start | Ensure you're in the project root and `mlflow.db` exists |
| `No module named 'ultralytics'` | Run `pip install -r requirements.txt` |
| `git push` rejected | Run `git pull --rebase` first, then `git push` |
| `.dvc` files show changes | Run `dvc checkout` to sync data with the current pointer files |

---

## License

This project is developed for educational purposes as part of the Resep.in recipe recommendation system.
