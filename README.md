# Bone Fracture Detection - Computer Vision Project

Complete computer vision pipeline for bone fracture detection with three modules: Image Processing, Classical Features, and Deep Learning.

## Project Structure

### Core Implementation
- `complete_all_modules.py` - Main pipeline (All 3 modules integrated)
- `src/bonefracture/` - Source code package
- `train_yolo_proper.py` - YOLO training script
- `realtime_yolo_detection.py` - Real-time detection
- `segnet.py` - Segmentation model

### Training on Colab
- `colab_yolo_training.ipynb` - YOLO training notebook (5-10 epochs)
- `colab_setup.ipynb` - General Colab setup

### Reports
- `research_reports/` - Module reports (Required)
- `results/` - Final reports

## Quick Start

### Local Training
```bash
pip install -r requirements.txt
python complete_all_modules.py
```

### Colab Training (Recommended)
1. Open `colab_yolo_training.ipynb` in VS Code
2. Install Colab extension
3. Connect to Colab and select GPU runtime
4. Run cells to train YOLO model

### Real-Time Detection
```bash
python realtime_yolo_detection.py --source webcam --model path/to/model.pt
```

## Project Requirements

See `PROJECT_REQUIREMENTS_ANALYSIS.md` for complete requirements compliance.

## Documentation


- `PROJECT_REQUIREMENTS_ANALYSIS.md` - Requirements compliance analysis

## Models

- **DenseNet-121**: Classification (74.56% accuracy)
- **YOLOv8**: Object detection (7 fracture types)
- **SegNet**: Segmentation

## GPU Requirements

- Minimum: 4 GB VRAM (YOLOv8n)
- Recommended: 8 GB VRAM (YOLOv8s)
- Optimal: 16+ GB VRAM (YOLOv8m/l)

For training, use Google Colab (free GPU) or local GPU.
