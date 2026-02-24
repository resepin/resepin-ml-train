# Resep.in — Food Ingredient Detection (ML Training)

This is the **machine learning training repository** for [Resep.in](https://github.com/resepin), a recipe recommendation app. It uses a **YOLOv8** object detection model to identify **53 food ingredients** from images, enabling the app to suggest recipes based on detected ingredients.

| Item | Detail |
|------|--------|
| Model | YOLOv8n (Ultralytics) |
| Classes | 53 food ingredients |
| Dataset | ~50,800 images (~30 GB) from 5 merged Roboflow sources |
| Framework | PyTorch + Ultralytics |
| Experiment Tracking | MLflow (local SQLite) |
| Dataset Versioning | DVC → Google Drive |
| Code Versioning | Git → GitHub |

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Dataset Download](#3-dataset-download)
4. [What `train_model.ipynb` Does](#4-what-train_modelipynb-does)
5. [How to Run Training](#5-how-to-run-training)
6. [Viewing Model Versioning (MLflow)](#6-viewing-model-versioning-mlflow)
7. [Model Promotion](#7-model-promotion)
8. [Project Structure](#8-project-structure)
9. [Detected Classes (53)](#9-detected-classes-53)
10. [How DVC, MLflow, and Google Drive Work Together](#10-how-dvc-mlflow-and-google-drive-work-together)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

Ensure the following are installed **before** starting:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.8+ (tested on 3.12) | Runtime |
| Git | 2.30+ | Version control |
| NVIDIA GPU | CUDA-capable | Training acceleration |
| CUDA Toolkit | 11.8 or 12.x | Must match PyTorch build |
| pip | Latest | Package installer |
| Google Account | — | Download dataset from Google Drive |
| Disk Space | ~35 GB free | Dataset + weights + outputs |
| RAM | 16 GB+ | Data loading during training |

> **GPU check:** Run `nvidia-smi` in a terminal. If you see your GPU name and driver version, you are ready.

---

## 2. Installation

```bash
# 1. Clone the repository
git clone https://github.com/resepin/resepin-ml-train.git
cd resepin-ml-train

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA (pick your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install DVC Google Drive support
pip install dvc-gdrive

# 6. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA GeForce RTX XXXX
```

---

## 3. Dataset Download

The dataset (~30 GB) is stored on **Google Drive** and managed by **DVC**.

### Google Drive Link

> **[https://drive.google.com/drive/folders/1hMKReeEKfiWFHtGGCQ9MmkVBsKMZ8Hs9](https://drive.google.com/drive/folders/1hMKReeEKfiWFHtGGCQ9MmkVBsKMZ8Hs9)**

### Download via DVC (recommended)

```bash
dvc pull
```

On first run, DVC opens your browser for Google authentication. After completion:

```
dataset_normalisasi_fix/          ← ~30 GB
├── data.yaml                     ← YOLO config (53 class names)
├── manifest.json                 ← Dataset fingerprint (SHA-256 hash)
├── train/images/                 ← 40,696 training images
├── train/labels/                 ← YOLO-format labels
├── valid/images/                 ← 5,088 validation images
├── valid/labels/
├── test/images/                  ← 5,088 test images
└── test/labels/

raw_dataset/                      ← 5 original Roboflow sources
yolov8n.pt                        ← Pre-trained YOLOv8 Nano weights
```

### Manual download (fallback)

If `dvc pull` fails, download directly from the [Google Drive folder](https://drive.google.com/drive/folders/1hMKReeEKfiWFHtGGCQ9MmkVBsKMZ8Hs9) and place the contents in the project root matching the structure above.

### Verify

```bash
dvc status    # Should show no changes
```

---

## 4. What `train_model.ipynb` Does

The notebook has **10 cells** across 4 stages:

### Stage 1 — Setup (Cells 1–3)

| Cell | Action |
|------|--------|
| 1 | Displays versioning overview (DVC, MLflow, Model Registry) |
| 2 | Installs Ultralytics (YOLOv8 framework) |
| 3 | Installs PyTorch with CUDA 11.8 support |

### Stage 2 — Verification (Cells 4–6)

| Cell | Action |
|------|--------|
| 4 | Imports YOLO, checks GPU availability, loads `yolov8n.pt` |
| 5 | GPU diagnostics: Python version, PyTorch version, CUDA test, tensor allocation test |
| 6 | Dataset statistics: counts images per split, prints totals and split ratios |

### Stage 3 — Training (Cell 7)

The core cell. Creates a `TrainingConfig` and calls `run_training(config)`, which automatically:

1. Loads the **dataset manifest** (SHA-256 fingerprint of all labels)
2. Starts an **MLflow run** and logs all hyperparameters
3. **Trains YOLOv8** on the dataset (50 epochs, batch=16, AdamW, lr=0.001)
4. Logs **epoch-by-epoch metrics** (mAP, precision, recall, loss) to MLflow
5. Saves **model weights** (`best.pt`, `last.pt`) as MLflow artifacts
6. Saves **training plots** (confusion matrix, PR curve, F1 curve)
7. Creates `run_metadata.json` linking run ID → dataset hash → git commit

### Stage 4 — Post-Training (Cells 8–10)

| Cell | Action |
|------|--------|
| 8 | Instructions for MLflow UI and model promotion |
| 9 | Promotes trained model to "production" (compares against current best on mAP50-95) |
| 10 | Lists all registered model versions with their metrics and status |

---

## 5. How to Run Training

### Option A: Notebook (recommended)

Open `train_model.ipynb` and run all cells top to bottom. Training takes ~2–6 hours depending on GPU.

```python
from train_versioned import run_training, TrainingConfig

config = TrainingConfig(
    model_name="yolov8n.pt",
    epochs=50,
    patience=3,
    batch=16,           # Reduce to 8 if GPU out of memory
    imgsz=640,
    optimizer="AdamW",
    lr0=0.001,
    device="0",         # "0" = first GPU
)

result = run_training(config)
```

### Option B: Command line

```bash
python train_versioned.py
```

### Output

Results are saved to `runs/detect/<run_name>/`:

```
runs/detect/v1.0.0_yolov8n_20260224_143000/
├── weights/best.pt          ← Best model (use for deployment)
├── weights/last.pt          ← Last epoch
├── results.csv              ← Per-epoch metrics
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── results.png              ← Training curves
├── args.yaml                ← Full config used
└── run_metadata.json        ← MLflow run ID + dataset hash
```

---

## 6. Viewing Model Versioning (MLflow)

Every training run is automatically tracked. To view experiments, metrics, and model versions:

### Launch the dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open **http://localhost:5000** in your browser.

### What you see

| Tab | Description |
|-----|-------------|
| **Runs table** | All training runs with hyperparameters and final metrics (sortable) |
| **Metric charts** | Click a run → epoch-by-epoch plots of loss, mAP, precision, recall |
| **Artifacts** | Browse/download model weights, confusion matrix, PR curves per run |
| **Compare** | Select 2+ runs → side-by-side metric comparison |
| **Model Registry** | Registered models with "production" / "staging" labels |
| **Tags** | Each run tagged with `dataset_version`, `dataset_hash`, `git_commit`, `model_arch` |

### Metrics tracked per run

| Metric | Description |
|--------|-------------|
| `mAP50` | Mean Average Precision @ IoU=0.50 **(primary)** |
| `mAP50_95` | Mean Average Precision @ IoU=0.50:0.95 |
| `precision` | Fraction of correct detections |
| `recall` | Fraction of actual objects detected |
| `train_box_loss` | Training bounding box loss |
| `train_cls_loss` | Training classification loss |
| `val_box_loss` | Validation bounding box loss |
| `val_cls_loss` | Validation classification loss |

### Quick check (without UI)

```bash
python -c "from mlflow_config import setup_mlflow; setup_mlflow()"
# Expected: MLflow experiment: resep-in-food-detection (ID: ...)
```

---

## 7. Model Promotion

### From notebook (Cell 9)

```python
from promote_model import promote_model
promote_model(run_id=result['run_id'], metric_name="mAP50_95", min_threshold=0.15)
```

### From command line

```bash
python promote_model.py list                     # List all versions
python promote_model.py promote <run_id> 0.15    # Promote if better
```

**How it works:** Candidate metric is compared against the current production model. Higher → promoted to production. Lower → saved as staging.

---

## 8. Project Structure

```
resepin-ml-train/
├── train_model.ipynb              # Training notebook (start here)
├── dataset.ipynb                  # Dataset merge/preprocessing notebook
│
├── train_versioned.py             # Training engine with MLflow integration
├── version_config.py              # Paths, class list, dataset version
├── dataset_manifest.py            # Dataset SHA-256 fingerprint generator
├── mlflow_config.py               # MLflow SQLite setup and helpers
├── log_results.py                 # YOLO results.csv → MLflow logger
├── promote_model.py               # Model registry: register, promote, list
├── dataset_pipeline.py            # Merges 5 Roboflow datasets → clean dataset
├── requirements.txt               # Python dependencies
│
├── dataset_normalisasi_fix/       # [DVC] Final dataset (~30 GB)
├── dataset_normalisasi_fix.dvc    # DVC pointer
├── raw_dataset/                   # [DVC] 5 original Roboflow datasets
├── raw_dataset.dvc                # DVC pointer
├── yolov8n.pt                     # [DVC] Pre-trained weights
├── yolov8n.pt.dvc                 # DVC pointer
│
├── runs/detect/                   # Training outputs
├── mlflow.db                      # MLflow experiment database
└── .dvc/config                    # DVC remote config (Google Drive)
```

---

## 9. Detected Classes (53)

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

## 10. How DVC, MLflow, and Google Drive Work Together

```
┌──────────────────────────────────────────────────────┐
│                   YOUR MACHINE                        │
│                                                       │
│   Code (.py, .ipynb)        Data (images, weights)    │
│         │                          │                  │
│    Git tracks                DVC tracks               │
│    full files                .dvc pointers             │
│         │                          │                  │
│         │       ┌──────────────────┘                  │
│         │       │                                     │
│         │  ┌────▼───────────────────────┐             │
│         │  │     MLflow (mlflow.db)      │             │
│         │  │  • Hyperparameters          │             │
│         │  │  • Metrics (per epoch)      │             │
│         │  │  • Model weights            │             │
│         │  │  • Dataset hash + git commit│             │
│         │  └────────────────────────────┘             │
│         │                          │                  │
└─────────┼──────────────────────────┼──────────────────┘
          │                          │
     git push                   dvc push
          │                          │
  ┌───────▼──────┐       ┌──────────▼─────────┐
  │   GitHub      │       │   Google Drive      │
  │ Code + .dvc   │       │ Images, labels,     │
  │ pointers      │       │ weights (~30 GB)    │
  └──────────────┘       └────────────────────┘
```

1. **Git + GitHub** — stores code and tiny `.dvc` pointer files (~200 bytes each)
2. **DVC + Google Drive** — stores actual large files. `dvc add` creates pointers, `dvc push` uploads, `dvc pull` downloads
3. **MLflow** — records every training run in `mlflow.db`: hyperparameters, epoch metrics, model weights, dataset hash, git commit

**Reproducibility:**

```bash
git checkout <commit>     # Exact code version
dvc checkout              # Exact dataset version
python train_versioned.py # Same hyperparameters → reproducible results
```

---

## 11. Troubleshooting

| Problem | Solution |
|---------|----------|
| `dvc pull` asks for authentication | Sign in with a Google account that has access to the Drive folder |
| `CUDA out of memory` | Reduce `batch` to 8 or 4 in `TrainingConfig` |
| `dvc push` SSL errors | Retry with `dvc push -j 1` (DVC resumes automatically) |
| `mlflow ui` won't start | Ensure you're in the project root and `mlflow.db` exists |
| `No module named 'ultralytics'` | Run `pip install -r requirements.txt` |
| `CUDA: False` | PyTorch CUDA version mismatch. Check `nvidia-smi`, reinstall with correct `--index-url` |
| `git push` rejected | Run `git pull --rebase` first |

---

## License

This project is developed for educational purposes as part of the Resep.in recipe recommendation system.
