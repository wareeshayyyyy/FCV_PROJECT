# GPU Fine-Tuning Guide - Google Colab

## 🚀 Quick Start: Run Fine-Tuning on GPU

### Step 1: Open Colab Notebook
1. Open `colab_setup.ipynb` in VS Code
2. Install **"Colab" extension** in VS Code (if not installed)
3. Click **"Connect to Colab"** button (top right)
4. Select **"New Colab Runtime"** → Choose **"GPU (T4)"**

### Step 2: Run Cells in Order

#### Cell 1: Install Dependencies
```python
# Already configured - just run it
```

#### Cell 2: Check GPU
```python
# Will show: ✅ GPU is ready!
# GPU Name: Tesla T4
# VRAM: 15.00 GB
```

#### Cell 3-4: Setup Dataset
**Option A: Upload from Computer**
- Click folder icon (left sidebar)
- Upload your dataset to `/content/bone_fracture_detection/data/archive/`

**Option B: Mount Google Drive**
- Run Cell 4 to mount Drive
- Copy dataset to Colab

**Option C: Clone from GitHub**
- Uncomment the git clone line in Cell 11
- Your project will be cloned automatically

#### Cell 11: Setup Project
```python
# Uncomment this line:
!cd /content && git clone https://github.com/wareeshayyyyy/FCV_PROJECT.git bone_fracture_detection
```

#### Cell 13: Configure Fine-Tuning
```python
# Automatically configures:
# - Dataset path
# - Batch size (16 for GPU)
# - Device settings
```

#### Cell 14: Run Fine-Tuning ⭐
```python
# This will run:
# Phase 1: 10 epochs (Classifier training)
# Phase 2: 10 epochs (Full fine-tuning)
# Total time: ~1-2 hours on GPU
```

#### Cell 15: Download Results
```python
# Downloads:
# - best_model_phase_1.pth
# - best_model_phase_2.pth
# - final_model_complete.pth
# - Training history plots
# - Results JSON
```

---

## 📊 Training Configuration (GPU)

- **Device**: GPU (Tesla T4, 16GB VRAM)
- **Batch Size**: 16 (auto-adjusted)
- **Phase 1**: 10 epochs, LR: 1e-3
- **Phase 2**: 10 epochs, LR: 1e-4 (backbone: 1e-5)
- **Total Time**: ~1-2 hours

---

## ✅ Expected Output

```
============================================================
DEVICE INFORMATION
============================================================
✅ CUDA Available: True
GPU Name: Tesla T4
VRAM: 15.00 GB
✅ GPU is ready for training!
============================================================

============================================================
PHASE 1 TRAINING
============================================================
Epoch 1/10
Training Metrics:
  Loss: 0.6234
  Accuracy: 0.7234 (72.34%)
Validation Metrics:
  Loss: 0.5123
  Accuracy: 0.7816 (78.16%)
...
```

---

## 🔧 Troubleshooting

### GPU Not Available?
1. Go to: **Runtime → Change runtime type → GPU (T4)**
2. Wait for GPU to initialize (~30 seconds)
3. Re-run Cell 2 to verify

### Dataset Not Found?
1. Check path: `/content/bone_fracture_detection/data/archive/bone fracture detection.v4-v4.yolov8`
2. Verify `data.yaml` exists
3. Check train/valid/test folders exist

### Out of Memory?
- Reduce batch size in Cell 13: `densenet_batch = 8` (instead of 16)
- Or use smaller model

---

## 📥 After Training

Models will be saved to:
- `/content/bone_fracture_detection/checkpoints/`
- Automatically downloaded to your computer
- Also saved to Google Drive (if mounted)

---

## 💡 Tips

1. **Keep Colab tab open** - Session disconnects after 90 min inactivity
2. **Save checkpoints** - Models auto-save every epoch
3. **Monitor progress** - Watch training metrics in real-time
4. **Download immediately** - Files deleted when session ends

---

**Ready to start? Open `colab_setup.ipynb` and connect to Colab!** 🚀

