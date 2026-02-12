# Resep.in - Food Ingredient Detection

Proyek deteksi bahan makanan menggunakan **YOLOv8** untuk sistem rekomendasi resep.

## Deskripsi

Proyek ini menggunakan model YOLOv8 untuk mendeteksi berbagai bahan makanan dari gambar. Hasil deteksi dapat digunakan untuk merekomendasikan resep berdasarkan bahan yang tersedia.

## Struktur Folder

```
resep.in/
├── dataset_normalisasi_fix/    # Dataset utama (sudah dinormalisasi)
│   ├── train/                  # Data training
│   ├── valid/                  # Data validasi
│   ├── test/                   # Data testing
│   └── data.yaml               # Konfigurasi dataset YOLO
├── raw dataset/                # Dataset mentah dari berbagai sumber
├── runs/detect/                # Output training & prediksi YOLO
├── dataset.ipynb               # Notebook preprocessing dataset
├── train_model.ipynb           # Notebook training model
└── yolov8n.pt                  # Pre-trained weights YOLOv8 Nano
```

## Kelas yang Dideteksi (59 Kelas)

| Bahan Mentah | Bahan Olahan | Buah |
|--------------|--------------|------|
| beef, pork, chicken | fried_chicken, fried_egg | apple, banana, kiwi |
| fish, shrimp, tuna | boiled_egg, fried_tofu | lemon, papaya, pineapple |
| egg, tofu, tempeh | fried_tempeh, meatball | kumquat |
| carrot, potato, onion | burger, pizza, spaghetti | |
| garlic, ginger, shallot | french_fries, donut | |
| tomato, cucumber, cabbage | chicken_nugget, steak | |
| bell_pepper, chili, broccoli | beef_rendang | |
| eggplant, cauliflower, corn | eggplant_balado | |
| pumpkin, chayote, bitter_gourd | | |
| bottle_gourd, mackerel, milkfish, tilapia | | |

## Requirements

- Python 3.8+
- PyTorch (dengan CUDA untuk GPU)
- Ultralytics

## Instalasi

```bash
# Install dependencies
pip install ultralytics --upgrade
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Penggunaan

### 1. Training Model

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='dataset_normalisasi_fix/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
```

### 2. Prediksi

```python
from ultralytics import YOLO

model = YOLO('runs/detect/training_results/weights/best.pt')
results = model.predict(source='path/to/image.jpg')
```

## Hasil Training

Output training tersimpan di `runs/detect/`:
- `training_results/` - Hasil training utama
- `weights/best.pt` - Model terbaik
- `weights/last.pt` - Model terakhir

## Lisensi

Proyek ini dibuat untuk keperluan edukasi dan pengembangan sistem rekomendasi resep.
