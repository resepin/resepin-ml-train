# Resep.in — Food Ingredient Detection (ML Training)

Machine learning training repository for **Resep.in**, a recipe recommendation app. A **YOLOv8n** object detection model identifies **53 food ingredients** from camera images so the app can suggest matching recipes.

## Current Production Model

| Metric | Value |
|--------|-------|
| **Model** | YOLOv8n (Ultralytics) — `best.pt` |
| **Dataset** | v1.0.0 — 44,235 images (30,965 train / 6,635 valid / 6,635 test) |
| **mAP50** | **0.8965** |
| **mAP50-95** | **0.6924** |
| **Precision** | 0.862 |
| **Recall** | 0.859 |
| **Training** | 50 epochs, AdamW, lr=0.001, batch=16, imgsz=640 |
| **Hardware** | NVIDIA RTX 5070 Laptop GPU — ~5 hours |
| **Run ID** | `2c4ffcc4cca044e09fe60c03fe3d3f72` |

## Stack

| Component | Tool | Role |
|-----------|------|------|
| Detection model | YOLOv8n (PyTorch + Ultralytics) | Train & infer |
| Experiment tracking | MLflow (local SQLite) | Hyperparams, metrics, artifacts |
| Model registry | MLflow Model Registry | Production / Staging promotion |
| Dataset versioning | DVC → Google Drive | ~30 GB images + labels |
| Code versioning | Git → GitHub | Code, configs, `.dvc` pointers |

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Dataset Download](#3-dataset-download)
4. [Training](#4-training)
5. [Model Promotion](#5-model-promotion)
6. [MLflow Dashboard](#6-mlflow-dashboard)
7. [Project Structure](#7-project-structure)
8. [Detected Classes (53)](#8-detected-classes-53)
9. [Architecture Overview](#9-architecture-overview)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.12 | Runtime |
| Git | 2.30+ | Version control |
| NVIDIA GPU | CUDA-capable | Training acceleration |
| CUDA Toolkit | 11.8 or 12.x | Must match PyTorch build |
| pip | Latest | Package installer |
| Google Account | — | Download dataset from Google Drive |
| Disk Space | ~35 GB free | Dataset + weights + outputs |
| RAM | 16 GB+ | Data loading during training |

> **GPU check:** Run `nvidia-smi` in a terminal. If you see your GPU name and driver version, you're ready.

---

## 2. Installation

```bash
# 1. Clone the repository
git clone https://github.com/Otniel1018/resepin-ml-train.git
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
dataset_normalisasi_fix/          ← ~30 GB (v1.0.0)
├── data.yaml                     ← YOLO config (53 class names)
├── manifest.json                 ← Dataset fingerprint (SHA-256 hash)
├── train/images/                 ← 30,965 training images
├── train/labels/                 ← YOLO-format labels
├── valid/images/                 ← 6,635 validation images
├── valid/labels/
├── test/images/                  ← 6,635 test images
└── test/labels/

raw_dataset/                      ← 5 original Roboflow sources
yolov8n.pt                        ← Pre-trained YOLOv8 Nano weights
```

**Source datasets** (merged & normalized in `dataset_pipeline.py`):

| # | Dataset | Source |
|---|---------|--------|
| 1 | Bahan-Makanan-Rumah-Tangga-1 | Roboflow |
| 2 | Food-Ingredient-Detection-1 | Roboflow |
| 3 | food-ingredients-1 | Roboflow |
| 4 | Indonesian-Food-1 | Roboflow |
| 5 | Recipe-Recommendation-System-2-1 | Roboflow |

### Manual download (fallback)

If `dvc pull` fails, download directly from the [Google Drive folder](https://drive.google.com/drive/folders/1hMKReeEKfiWFHtGGCQ9MmkVBsKMZ8Hs9) and place the contents in the project root.

### Verify

```bash
dvc status    # Should show no changes
```

---

## 4. Training

### Option A: Notebook (recommended)

Open `train_model.ipynb` and run all cells top to bottom:

| Cells | Stage | What it does |
|-------|-------|-------------|
| 1–3 | Setup | Install Ultralytics + PyTorch CUDA |
| 4–6 | Verify | Import model, GPU diagnostics, dataset statistics |
| 7 | Train | Run versioned training with MLflow tracking |
| 8–10 | Post | Promote model, list versions |

```python
from train_versioned import TrainingConfig, run_training

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

### What happens during training

1. Loads the **dataset manifest** (SHA-256 fingerprint of all labels)
2. Starts an **MLflow run** and logs all hyperparameters
3. **Trains YOLOv8** with the configured settings
4. Logs **epoch-by-epoch metrics** (mAP, precision, recall, loss) to MLflow
5. Saves **model weights** (`best.pt`, `last.pt`) as MLflow artifacts
6. Saves **training plots** (confusion matrix, PR curve, F1 curve)
7. Creates `run_metadata.json` linking run ID → dataset hash → git commit

### Output

Results are saved to `runs/detect/<run_name>/`:

```
runs/detect/v1.0.0_yolov8n_20260225_092135/
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

## 5. Model Promotion

After training, promote the best model to production:

### From notebook (Cell 9)

```python
from promote_model import promote_model

promote_model(
    run_id=result["run_id"],
    metric_name="mAP50_95",
    min_threshold=0.15,
    description=f"Dataset v{result['run_name']}",
)
```

### From command line

```bash
python promote_model.py list                     # List all versions
python promote_model.py promote <run_id> 0.15    # Promote if better
```

**How it works:**
- Candidate metric is compared against the current production model
- Higher → promoted to **production** (alias updated in MLflow Model Registry)
- Lower → saved as **staging**
- Below `min_threshold` → skipped entirely

---

## 6. MLflow Dashboard

### Launch

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open **http://localhost:5000** in your browser.

### What you see

| Tab | Description |
|-----|-------------|
| **Runs table** | All training runs with hyperparameters and final metrics |
| **Metric charts** | Click a run → epoch-by-epoch mAP, loss, precision, recall |
| **Artifacts** | Browse/download model weights, confusion matrix, PR curves |
| **Compare** | Select 2+ runs → side-by-side metrics |
| **Model Registry** | Registered models with production / staging aliases |

### Metrics tracked

| Metric | Description |
|--------|-------------|
| `mAP50` | Mean Average Precision @ IoU=0.50 |
| `mAP50-95` | Mean Average Precision @ IoU=0.50:0.95 **(primary)** |
| `precision` | Fraction of correct detections |
| `recall` | Fraction of actual objects detected |
| `train/val box_loss` | Bounding box regression loss |
| `train/val cls_loss` | Classification loss |
| `train/val dfl_loss` | Distribution focal loss |

### Quick check (without UI)

```bash
python -c "from promote_model import list_models; list_models()"
```

---

## 7. Project Structure

```
resepin-ml-train/
├── train_model.ipynb              # Training notebook (start here)
├── dataset.ipynb                  # Dataset merge/preprocessing notebook
│
├── train_versioned.py             # Training engine with MLflow integration
├── version_config.py              # Paths, class list, dataset version
├── dataset_manifest.py            # Dataset SHA-256 fingerprint generator
├── dataset_pipeline.py            # Merges 5 Roboflow datasets → clean dataset
├── mlflow_config.py               # MLflow SQLite setup and helpers
├── log_results.py                 # YOLO results.csv → MLflow logger
├── promote_model.py               # Model registry: register, promote, list
├── requirements.txt               # Python dependencies
│
├── dataset_normalisasi_fix/       # [DVC] Final dataset (~30 GB)
├── dataset_normalisasi_fix.dvc    # DVC pointer
├── raw_dataset/                   # [DVC] 5 original Roboflow datasets
├── raw_dataset.dvc                # DVC pointer
├── yolov8n.pt                     # [DVC] Pre-trained YOLOv8n weights
├── yolov8n.pt.dvc                 # DVC pointer
│
├── runs/detect/                   # Training outputs (per run)
├── mlflow.db                      # MLflow experiment database
└── .dvc/config                    # DVC remote config (Google Drive)
```

---

## 8. Detected Classes (53)

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

## 9. Architecture Overview

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

**Reproducibility:**

```bash
git checkout <commit>     # Exact code version
dvc checkout              # Exact dataset version
python train_versioned.py # Same hyperparameters → reproducible results
```

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `dvc pull` asks for authentication | Sign in with a Google account that has access to the Drive folder |
| `CUDA out of memory` | Reduce `batch` to 8 or 4 in `TrainingConfig` |
| `dvc push` SSL errors | Retry with `dvc push -j 1` (DVC resumes automatically) |
| `mlflow ui` won't start | Ensure you're in the project root and `mlflow.db` exists |
| `No module named 'ultralytics'` | Run `pip install -r requirements.txt` |
| `CUDA: False` | PyTorch CUDA version mismatch. Check `nvidia-smi`, reinstall with correct `--index-url` |
| `git push` rejected | Run `git pull --rebase` first |
| `promote_model` shows 0.0 metrics | Restart notebook kernel to reload fixed modules |

---

## License

This project is developed for educational purposes as part of the Resep.in recipe recommendation system.
