"""
Proper YOLO Training Script for Bone Fracture Detection
Trains YOLOv8 model on bone fracture dataset with 5-10 epochs
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
import json
from datetime import datetime


class YOLOTrainer:
    """YOLO Model Trainer for Bone Fracture Detection"""
    
    def __init__(self, 
                 data_yaml_path,
                 model_size='n',  # n, s, m, l, x
                 epochs=10,
                 imgsz=640,
                 batch=16,
                 device=None):
        """
        Initialize YOLO trainer
        
        Args:
            data_yaml_path: Path to data.yaml file
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            epochs: Number of training epochs (5-10)
            imgsz: Image size for training
            batch: Batch size
            device: Device to use (None for auto-detect)
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Model name
        self.model_name = f'yolov8{model_size}.pt'
        
        # Output directory
        self.output_dir = Path('yolo_training_results')
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("YOLO BONE FRACTURE DETECTION - TRAINING SETUP")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Image Size: {self.imgsz}")
        print(f"Batch Size: {self.batch}")
        print(f"Data Config: {self.data_yaml_path}")
        print("="*80)
        
        # Check GPU
        if self.device == 'cuda':
            print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n⚠️  No GPU detected - Training will be slower on CPU")
    
    def check_gpu_requirements(self):
        """Check and display GPU requirements"""
        print("\n" + "="*80)
        print("GPU REQUIREMENTS ANALYSIS")
        print("="*80)
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n✅ Current GPU: {gpu_name}")
            print(f"   Available Memory: {gpu_memory:.2f} GB")
            
            # Recommended requirements
            print("\n📊 Recommended GPU Requirements:")
            print("   Minimum: 4 GB VRAM (YOLOv8n, batch=8)")
            print("   Recommended: 8 GB VRAM (YOLOv8s, batch=16)")
            print("   Optimal: 16+ GB VRAM (YOLOv8m/l, batch=32+)")
            
            # Estimate memory usage
            estimated_memory = {
                'n': 2.5,
                's': 4.0,
                'm': 8.0,
                'l': 12.0,
                'x': 20.0
            }
            
            est_mem = estimated_memory.get(self.model_size, 4.0)
            print(f"\n   Estimated Memory Usage (YOLOv8{self.model_size}): ~{est_mem} GB")
            
            if gpu_memory < 4:
                print("\n⚠️  WARNING: Low GPU memory. Consider:")
                print("   - Using YOLOv8n (nano) model")
                print("   - Reducing batch size to 8 or 4")
                print("   - Reducing image size to 416")
            elif gpu_memory < 8:
                print("\n✅ GPU memory is adequate for YOLOv8n/s")
            else:
                print("\n✅ GPU memory is excellent for larger models")
        else:
            print("\n⚠️  CPU Training Mode")
            print("   Training will be significantly slower")
            print("   Recommended: Use GPU for faster training")
            print("   Estimated time: 2-5 hours per epoch (CPU)")
            print("   Estimated time: 10-30 minutes per epoch (GPU)")
    
    def train(self):
        """Train YOLO model"""
        print("\n" + "="*80)
        print("STARTING YOLO TRAINING")
        print("="*80)
        
        # Load pretrained model
        print(f"\n📥 Loading pretrained model: {self.model_name}")
        model = YOLO(self.model_name)
        
        # Verify data.yaml exists
        if not os.path.exists(self.data_yaml_path):
            print(f"❌ Error: Data config file not found: {self.data_yaml_path}")
            return None
        
        # Read data config
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"\n📊 Dataset Configuration:")
        print(f"   Classes: {data_config.get('nc', 'N/A')}")
        print(f"   Class Names: {data_config.get('names', [])}")
        
        # Training arguments
        train_args = {
            'data': self.data_yaml_path,
            'epochs': self.epochs,
            'imgsz': self.imgsz,
            'batch': self.batch,
            'device': self.device,
            'project': str(self.output_dir),
            'name': f'yolov8{self.model_size}_bone_fracture',
            'save': True,
            'save_period': 2,  # Save checkpoint every 2 epochs
            'val': True,  # Validate during training
            'plots': True,  # Generate training plots
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,  # Close mosaic augmentation in last 10 epochs
            'resume': False,
            'amp': True,  # Automatic Mixed Precision (faster training)
            'fraction': 1.0,  # Use 100% of dataset
            'profile': False,
            'freeze': None,  # Don't freeze layers
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Class loss gain
            'dfl': 1.5,  # DFL loss gain
            'pose': 12.0,  # Pose loss gain
            'kobj': 1.0,  # Keypoint obj loss gain
            'label_smoothing': 0.0,
            'nbs': 64,  # Nominal batch size
            'hsv_h': 0.015,  # Image HSV-Hue augmentation
            'hsv_s': 0.7,  # Image HSV-Saturation augmentation
            'hsv_v': 0.4,  # Image HSV-Value augmentation
            'degrees': 0.0,  # Image rotation (+/- deg)
            'translate': 0.1,  # Image translation (+/- fraction)
            'scale': 0.5,  # Image scale (+/- gain)
            'shear': 0.0,  # Image shear (+/- deg)
            'perspective': 0.0,  # Image perspective (+/- fraction)
            'flipud': 0.0,  # Image flip up-down (probability)
            'fliplr': 0.5,  # Image flip left-right (probability)
            'mosaic': 1.0,  # Image mosaic (probability)
            'mixup': 0.0,  # Image mixup (probability)
            'copy_paste': 0.0,  # Segment copy-paste (probability)
        }
        
        print(f"\n🚀 Starting training...")
        print(f"   This may take {'10-30 minutes per epoch' if self.device == 'cuda' else '2-5 hours per epoch'}")
        
        try:
            # Train the model
            results = model.train(**train_args)
            
            # Get best model path
            best_model_path = self.output_dir / f'yolov8{self.model_size}_bone_fracture' / 'weights' / 'best.pt'
            last_model_path = self.output_dir / f'yolov8{self.model_size}_bone_fracture' / 'weights' / 'last.pt'
            
            print("\n" + "="*80)
            print("TRAINING COMPLETE!")
            print("="*80)
            
            if best_model_path.exists():
                print(f"\n✅ Best model saved: {best_model_path}")
            if last_model_path.exists():
                print(f"✅ Last model saved: {last_model_path}")
            
            # Save training summary
            self.save_training_summary(results, best_model_path)
            
            return {
                'best_model': str(best_model_path),
                'last_model': str(last_model_path),
                'results': results,
                'metrics': self.extract_metrics(results)
            }
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_metrics(self, results):
        """Extract training metrics"""
        metrics = {}
        
        try:
            # Get metrics from results
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            elif hasattr(results, 'metrics'):
                metrics = results.metrics
            
            # Try to read from results file
            results_file = self.output_dir / f'yolov8{self.model_size}_bone_fracture' / 'results.csv'
            if results_file.exists():
                import pandas as pd
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    metrics = {
                        'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                        'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(last_row.get('metrics/precision(B)', 0)),
                        'recall': float(last_row.get('metrics/recall(B)', 0)),
                        'final_epoch': int(last_row.get('epoch', self.epochs))
                    }
        except Exception as e:
            print(f"Warning: Could not extract all metrics: {e}")
        
        return metrics
    
    def save_training_summary(self, results, model_path):
        """Save training summary to JSON"""
        summary = {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'epochs': self.epochs,
            'image_size': self.imgsz,
            'batch_size': self.batch,
            'device': self.device,
            'best_model_path': str(model_path),
            'training_date': datetime.now().isoformat(),
            'metrics': self.extract_metrics(results)
        }
        
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n📄 Training summary saved: {summary_path}")
        
        # Print metrics
        metrics = summary['metrics']
        if metrics:
            print("\n" + "="*80)
            print("TRAINING METRICS")
            print("="*80)
            print(f"mAP50: {metrics.get('mAP50', 'N/A'):.4f}" if isinstance(metrics.get('mAP50'), (int, float)) else f"mAP50: {metrics.get('mAP50', 'N/A')}")
            print(f"mAP50-95: {metrics.get('mAP50-95', 'N/A'):.4f}" if isinstance(metrics.get('mAP50-95'), (int, float)) else f"mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
            print(f"Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"Precision: {metrics.get('precision', 'N/A')}")
            print(f"Recall: {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall', (int, float))) else f"Recall: {metrics.get('recall', 'N/A')}")
            print("="*80)


def main():
    """Main training function"""
    # Configuration
    DATA_YAML = r'data\archive\bone fracture detection.v4-v4.yolov8\data.yaml'
    
    # Fix data.yaml paths if needed
    data_yaml_path = Path(DATA_YAML)
    if not data_yaml_path.exists():
        print(f"Error: Data YAML not found at {DATA_YAML}")
        print("Please check the path to your data.yaml file")
        return
    
    # Update data.yaml with correct paths
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Fix paths to be absolute paths (YOLO needs absolute paths to avoid duplication)
    base_dir = data_yaml_path.parent.resolve()  # Use resolve() to get absolute path
    # Use absolute paths and normalize
    train_path = str(base_dir / 'train' / 'images').replace('\\', '/')
    val_path = str(base_dir / 'valid' / 'images').replace('\\', '/')
    test_path = str(base_dir / 'test' / 'images').replace('\\', '/')
    
    # Also update the 'path' field in data_config to point to base_dir
    data_config['path'] = str(base_dir).replace('\\', '/')
    
    data_config['train'] = train_path
    data_config['val'] = val_path
    data_config['test'] = test_path
    
    # Verify paths exist
    if not Path(train_path).exists():
        print(f"⚠️  Warning: Train path does not exist: {train_path}")
    if not Path(val_path).exists():
        print(f"⚠️  Warning: Validation path does not exist: {val_path}")
    if not Path(test_path).exists():
        print(f"⚠️  Warning: Test path does not exist: {test_path}")
    
    # Save updated config
    updated_yaml = data_yaml_path.parent / 'data_updated.yaml'
    with open(updated_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"✅ Updated paths:")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    print(f"   Test: {test_path}")
    
    print(f"✅ Updated data config saved to: {updated_yaml}")
    
    # Initialize trainer
    trainer = YOLOTrainer(
        data_yaml_path=str(updated_yaml),
        model_size='n',  # Start with nano for faster training, change to 's', 'm', 'l', 'x' for better accuracy
        epochs=10,  # 5-10 epochs as requested
        imgsz=640,
        batch=16,  # Adjust based on GPU memory
        device=None  # Auto-detect
    )
    
    # Check GPU requirements
    trainer.check_gpu_requirements()
    
    # Train model
    results = trainer.train()
    
    if results:
        print("\n✅ Training completed successfully!")
        print(f"\n📁 Model saved to: {results['best_model']}")
        print(f"\n💡 To use for real-time detection:")
        print(f"   python realtime_yolo_detection.py --source webcam --model {results['best_model']}")
    else:
        print("\n❌ Training failed. Please check the error messages above.")


if __name__ == '__main__':
    main()

