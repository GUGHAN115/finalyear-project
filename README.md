# Robust Deepfake Detection using Multi-Level Feature Extraction

A PyTorch + OpenCV deep learning system for detecting AI-generated face manipulations with hierarchical forensic feature analysis.

---

## Architecture Overview

```
Input → Haar Cascade Face Detection (15% margin) → 224×224 crop
    ↓
ResNet-50 Backbone (ImageNet pretrained)
    ├── layer1 (256ch, 56×56)  → Low-level: edges, blending boundaries
    ├── layer2 (512ch, 28×28)  → Mid-level: facial geometry, landmarks
    └── layer4 (2048ch, 7×7)  → High-level: semantics, illumination
    ↓
Global Average Pooling × 3
    ↓
Linear Projections → [256, 256, 512]
    ↓
Concatenate → 1024-d Forensic Vector
    ↓
BN → FC(512) → ReLU → Dropout(0.4) → FC(2) → Softmax
    ↓
REAL / FAKE + Confidence Score
```

---

## Project Structure

```
finalyear project/
├── app.py               # Streamlit interactive dashboard
├── train.py             # Training script with AdamW + cosine LR
├── requirements.txt     # Python dependencies
├── src/
│   ├── __init__.py
│   ├── model.py         # DeepfakeDetector architecture
│   ├── preprocessor.py  # OpenCV Haar Cascade preprocessing
│   └── inference.py     # Image & video inference engine
└── checkpoints/
    └── best_model.pth   # Trained weights (place here after training)
```

---

## Installation

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# For CUDA GPU support (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Running the Dashboard

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Image Mode
1. Upload a JPG/PNG/WEBP image
2. OpenCV detects the face ROI (15% expanded bbox)
3. Inference returns REAL/FAKE verdict with confidence
4. Bounding box overlaid on the original image
5. Forensic feature heatmap and stage-contribution bars displayed

### Video Mode
1. Upload an MP4/AVI/MOV video
2. System samples N equidistant frames (configurable via sidebar)
3. Per-frame inference aggregated via mean P(FAKE)
4. Temporal timeline chart shows anomaly peaks
5. Full per-frame breakdown table available

---

## Training

### Dataset Layout
```
data/
├── train/
│   ├── real/   ← authentic face images
│   └── fake/   ← manipulated face images
└── val/
    ├── real/
    └── fake/
```

Compatible datasets:
- **FaceForensics++** (FF++) — most widely used benchmark
- **Celeb-DF v2** — high-quality celebrity deepfakes
- **DFDC** (DeepFake Detection Challenge) dataset

### Start Training
```bash
python train.py \
    --data_dir   data/ \
    --output_dir checkpoints/ \
    --epochs     20 \
    --batch_size 32 \
    --lr         1e-4
```

Training features:
- AdamW optimiser with weight decay `1e-4`
- Cosine annealing LR schedule (T_max = epochs)
- Mixed-precision training (FP16 AMP on CUDA)
- Label smoothing `0.1`
- Gradient clipping (max norm = 1.0)
- Best checkpoint saved automatically by val accuracy
- Periodic checkpoint every 5 epochs
- CSV metric log at `checkpoints/metrics.csv`

---

## Model Details

| Component | Details |
|-----------|---------|
| Backbone | ResNet-50, ImageNet-pretrained |
| Input size | 224 × 224 × 3 |
| Stage 1 features | 256-d (from layer1) |
| Stage 2 features | 256-d (projected from 512-ch layer2) |
| Stage 4 features | 512-d (projected from 2048-ch layer4) |
| Fused vector | 1,024-d |
| Classifier head | BN → FC(512) → ReLU → Dropout(0.4) → FC(2) |
| Parameters | ~25.6M total |
| Output | Softmax probability: [P(REAL), P(FAKE)] |

---

## Python API

```python
from src.inference import load_model, infer_image, infer_video
import cv2

# Load model
model, device = load_model(checkpoint_path="checkpoints/best_model.pth")

# --- Image inference ---
img = cv2.imread("face.jpg")
result = infer_image(img, model, device)

print(result["label"])       # "REAL" or "FAKE"
print(result["confidence"])  # e.g. 0.9342
print(result["prob_fake"])   # e.g. 0.0658
print(result["bbox"])        # (x1, y1, x2, y2)

# --- Video inference ---
vid_result = infer_video("video.mp4", model, device, n_frames=32)

print(vid_result["verdict"])           # "REAL" or "FAKE"
print(vid_result["mean_fake_prob"])    # mean P(FAKE) across frames
print(vid_result["peak_fake_prob"])    # highest P(FAKE) in any frame
print(vid_result["peak_frame_index"])  # frame with highest anomaly
```

---

## References

1. He, K. et al. — *Deep Residual Learning for Image Recognition* (CVPR 2016)
2. Rössler, A. et al. — *FaceForensics++* (ICCV 2019)
3. Li, Y. et al. — *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics* (CVPR 2020)
4. Dolhansky, B. et al. — *The DeepFake Detection Challenge Dataset* (2020)
5. Zhao, H. et al. — *Multi-attentional Deepfake Detection* (CVPR 2021)
