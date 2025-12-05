# Project Structure - Bone Fracture Detection

## Core Training Files (Module 3)

### Fine-Tuning (DenseNet-121)
- **`run_complete_training.py`** - Main fine-tuning script
  - Two-phase training (frozen → fine-tuned)
  - 10 epochs per phase
  - Differential learning rates
  - Complete metrics and visualization

### YOLO Object Detection
- **`train_yolo_proper.py`** - YOLO training script
  - YOLOv8 training for bone fracture detection
  - 5-10 epochs
  - Batch size and image size configuration
  
- **`realtime_yolo_detection.py`** - Real-time YOLO detection
  - Webcam/video/image detection
  - FPS counter
  - Visual annotations

### Supporting Training Files
- **`src/bonefracture/train_full.py`** - Package training module
- **`src/bonefracture/train.py`** - Basic training utilities

## Module Files

### Module 1: Image Preprocessing
- **`complete_all_modules.py`** - Complete pipeline (all modules)

### Module 2: Classical Features
- **`src/bonefracture/advanced_features.py`** - SIFT, SURF, HOG, BoVW, Geometric/Temporal

### Module 3: Deep Learning
- **`src/bonefracture/model.py`** - DenseNet-121 model
- **`segnet.py`** - SegNet segmentation
- **`src/bonefracture/utils/gradcam_example.py`** - Grad-CAM explainability

## Colab Notebooks
- **`colab_yolo_training.ipynb`** - YOLO training on Colab
- **`colab_setup.ipynb`** - General Colab setup

## Reports
- **`research_reports/`** - All module reports
- **`results/`** - Final reports and visualizations

