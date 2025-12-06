# YOLO Training & Real-Time Detection Guide

## 📋 Overview

### Two Different Models:

1. **DenseNet-121** (`run_complete_training.py`)
   - **Purpose**: Image Classification (Normal vs Fractured)
   - **Output**: Single label per image
   - **NOT for object detection**

2. **YOLOv8** (`train_yolo_proper.py`)
   - **Purpose**: Object Detection (Detects fracture locations in images)
   - **Output**: Bounding boxes + class labels
   - **✅ FOR REAL-TIME OBJECT DETECTION**

---

## 🎯 For Real-Time Object Detection:

### Step 1: Train YOLO Model

```bash
python train_yolo_proper.py
```

**What it does:**
- Trains YOLOv8 on bone fracture dataset
- Detects 7 fracture types:
  - Elbow positive
  - Fingers positive
  - Forearm fracture
  - Humerus fracture
  - Humerus
  - Shoulder fracture
  - Wrist positive
- Saves model to: `yolo_training_results/yolov8n_bone_fracture/weights/best.pt`

**Training Parameters:**
- Model: YOLOv8n (nano) - fast, good for real-time
- Epochs: 10
- Image Size: 640x640
- Batch Size: 16

---

### Step 2: Run Real-Time Detection

```bash
# Webcam detection
python realtime_yolo_detection.py --source webcam

# Video file
python realtime_yolo_detection.py --source video.mp4

# Image file
python realtime_yolo_detection.py --source image.jpg

# With custom model
python realtime_yolo_detection.py --source webcam --model yolo_training_results/yolov8n_bone_fracture/weights/best.pt
```

**Features:**
- ✅ Real-time FPS counter
- ✅ Bounding boxes with labels
- ✅ Confidence scores
- ✅ Color-coded fracture types
- ✅ Auto-detects trained model

---

## 🔄 Complete Workflow

### 1. Train YOLO Model (First Time)

```bash
# Train on your dataset
python train_yolo_proper.py
```

**Output:**
- Trained model: `yolo_training_results/yolov8n_bone_fracture/weights/best.pt`
- Training metrics and plots

### 2. Real-Time Detection

```bash
# The script auto-detects the trained model
python realtime_yolo_detection.py --source webcam
```

**Or specify model path:**
```bash
python realtime_yolo_detection.py --source webcam --model yolo_training_results/yolov8n_bone_fracture/weights/best.pt
```

---

## 📊 Model Comparison

| Feature | DenseNet-121 | YOLOv8 |
|---------|-------------|--------|
| **Task** | Classification | Object Detection |
| **Output** | Normal/Fractured | Bounding boxes + classes |
| **Real-time** | ❌ No | ✅ Yes |
| **Detects location** | ❌ No | ✅ Yes |
| **Multiple fractures** | ❌ No | ✅ Yes |
| **Training file** | `run_complete_training.py` | `train_yolo_proper.py` |
| **Detection file** | N/A | `realtime_yolo_detection.py` |

---

## 🚀 Quick Start for Real-Time Detection

### Option 1: Use Pretrained YOLOv8 (General Objects)
```bash
python realtime_yolo_detection.py --source webcam
```
⚠️ **Note**: This detects general objects (people, cars, etc.), NOT bone fractures!

### Option 2: Train Custom Model (Bone Fractures)
```bash
# Step 1: Train
python train_yolo_proper.py

# Step 2: Detect
python realtime_yolo_detection.py --source webcam
```

---

## 📁 File Structure

```
bone_fracture_densenet/
├── train_yolo_proper.py          # ✅ YOLO Training (Object Detection)
├── realtime_yolo_detection.py    # ✅ Real-Time Detection
├── run_complete_training.py      # DenseNet Training (Classification)
└── yolo_training_results/        # Trained YOLO models
    └── yolov8n_bone_fracture/
        └── weights/
            └── best.pt           # ✅ Use this for detection
```

---

## 💡 Tips

1. **For Real-Time**: Use YOLOv8n (nano) - fastest
2. **For Better Accuracy**: Train YOLOv8s or YOLOv8m
3. **GPU Recommended**: Real-time detection works on CPU but slower
4. **Webcam**: Press 'q' to quit detection

---

## 🎯 Summary

- **For Real-Time Object Detection**: Use `train_yolo_proper.py` + `realtime_yolo_detection.py`
- **For Image Classification**: Use `run_complete_training.py` (DenseNet-121)

**You need YOLO for real-time object detection!** 🎯

