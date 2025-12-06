"""
COMPLETE TRAINING SCRIPT - All Models with Fine-tuning
Runs full training with updated hyperparameters and provides complete metrics
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.append('src')
from bonefracture.bone_yolo_dataset import create_dataloaders, calculate_class_weights


class TrainingConfig:
    """Optimized training configuration"""
    DATASET_ROOT = r'data\archive\bone fracture detection.v4-v4.yolov8'
    CHECKPOINT_DIR = './checkpoints'
    OUTPUT_DIR = './training_results'
    
    # Model parameters
    NUM_CLASSES = 2
    PRETRAINED = True
    
    # Training parameters (OPTIMIZED FOR FINE-TUNING)
    BATCH_SIZE = 16
    NUM_EPOCHS_PHASE1 = 10  # Phase 1: Classifier training
    NUM_EPOCHS_PHASE2 = 10  # Phase 2: Full fine-tuning
    
    # Learning rates (OPTIMIZED FOR FINE-TUNING)
    LEARNING_RATE_PHASE1 = 1e-3  # Higher LR for classifier
    LEARNING_RATE_PHASE2 = 1e-4  # Lower LR for fine-tuning
    LEARNING_RATE_BACKBONE = 1e-5  # Even lower for backbone
    WEIGHT_DECAY = 1e-4
    
    # Hyperparameters
    DROPOUT_RATE = 0.5
    MOMENTUM = 0.9
    BETA1 = 0.9
    BETA2 = 0.999
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0 if not torch.cuda.is_available() else 4
    PIN_MEMORY = torch.cuda.is_available()
    
    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001
    
    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


class OptimizedBoneFractureClassifier(nn.Module):
    """Optimized DenseNet-121 with fine-tuned architecture"""
    
    def __init__(self, num_classes=2, pretrained=True, freeze_layers=True):
        super(OptimizedBoneFractureClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Layer freezing strategy
        if freeze_layers:
            # Freeze early layers
            for param in self.densenet.features.parameters():
                param.requires_grad = False
            
            # Unfreeze last dense block for fine-tuning
            try:
                for param in self.densenet.features.denseblock4.parameters():
                    param.requires_grad = True
                for param in self.densenet.features.norm5.parameters():
                    param.requires_grad = True
            except:
                pass
        
        # Optimized classifier head
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier with proper bias"""
        for m in self.densenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.densenet(x)
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_parameter_groups(self):
        """Get parameter groups for differential learning rates"""
        backbone_params = []
        classifier_params = []
        
        # Backbone parameters (features)
        for param in self.densenet.features.parameters():
            if param.requires_grad:
                backbone_params.append(param)
        
        # Classifier parameters
        for param in self.densenet.classifier.parameters():
            if param.requires_grad:
                classifier_params.append(param)
        
        return backbone_params, classifier_params


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Training epoch with detailed metrics"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc.item(), all_preds, all_labels


def validate(model, val_loader, criterion, device):
    """Validation with detailed metrics"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    
    return epoch_loss, epoch_acc.item(), all_preds, all_labels, all_probs


def train_model_complete(model, train_loader, val_loader, test_loader, config, phase_name='Phase 1'):
    """Complete training with all metrics"""
    print(f"\n{'='*80}")
    print(f"{phase_name.upper()} TRAINING")
    print(f"{'='*80}")
    
    # Loss function
    class_weights = calculate_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    
    # Optimizer with updated parameters
    if phase_name == 'Phase 1':
        # Phase 1: Only classifier parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            trainable_params,
            lr=config.LEARNING_RATE_PHASE1,
            weight_decay=config.WEIGHT_DECAY,
            betas=(config.BETA1, config.BETA2)
        )
        num_epochs = config.NUM_EPOCHS_PHASE1
        print(f"  Optimizer: Adam (Single LR: {config.LEARNING_RATE_PHASE1:.6f})")
    else:
        # Phase 2: Differential learning rates for fine-tuning
        backbone_params, classifier_params = model.get_parameter_groups()
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': config.LEARNING_RATE_BACKBONE, 'weight_decay': config.WEIGHT_DECAY},
            {'params': classifier_params, 'lr': config.LEARNING_RATE_PHASE2, 'weight_decay': config.WEIGHT_DECAY}
        ], betas=(config.BETA1, config.BETA2))
        num_epochs = config.NUM_EPOCHS_PHASE2
        print(f"  Optimizer: Adam (Differential LR)")
        print(f"    Backbone LR: {config.LEARNING_RATE_BACKBONE:.6f}")
        print(f"    Classifier LR: {config.LEARNING_RATE_PHASE2:.6f}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_acc = 0.0
    best_model_wts = None
    patience_counter = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Frozen Layers: {not any(p.requires_grad for p in model.densenet.features.parameters()) if phase_name == 'Phase 1' else 'None'}")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config
        )
        
        # Validation
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Learning rate (handle multiple param groups)
        if len(optimizer.param_groups) > 1:
            current_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
            classifier_lr = current_lr
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Print metrics
        print(f"\nTraining Metrics:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"\nHyperparameters:")
        if len(optimizer.param_groups) > 1:
            print(f"  Backbone Learning Rate: {current_lr:.6f}")
            print(f"  Classifier Learning Rate: {classifier_lr:.6f}")
        else:
            print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Weight Decay: {config.WEIGHT_DECAY}")
        print(f"  Optimizer: Adam (β1={config.BETA1}, β2={config.BETA2})")
        print(f"  Dropout: {config.DROPOUT_RATE}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict().copy()
            patience_counter = 0
            
            save_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'best_model_{phase_name.replace(" ", "_").lower()}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'learning_rate': current_lr,
                'config': config.__dict__
            }, save_path)
            print(f"\n✅ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, config.DEVICE
    )
    
    print(f"\nTest Metrics:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Calculate detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(test_labels, test_preds, average='weighted')
    recall = recall_score(test_labels, test_preds, average='weighted')
    f1 = f1_score(test_labels, test_preds, average='weighted')
    cm = confusion_matrix(test_labels, test_preds)
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'model': model,
        'best_val_acc': best_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'history': history,
        'test_predictions': test_preds,
        'test_labels': test_labels,
        'test_probs': test_probs
    }


def plot_training_history(history_phase1, history_phase2, config):
    """Plot complete training history"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs_phase1 = range(1, len(history_phase1['train_loss']) + 1)
    epochs_phase2 = range(1, len(history_phase2['train_loss']) + 1)
    epochs_phase2_adj = [e + len(epochs_phase1) for e in epochs_phase2]
    
    # Loss plot
    axes[0, 0].plot(epochs_phase1, history_phase1['train_loss'], 'b-', label='Phase 1 Train', linewidth=2)
    axes[0, 0].plot(epochs_phase1, history_phase1['val_loss'], 'b--', label='Phase 1 Val', linewidth=2)
    axes[0, 0].plot(epochs_phase2_adj, history_phase2['train_loss'], 'r-', label='Phase 2 Train', linewidth=2)
    axes[0, 0].plot(epochs_phase2_adj, history_phase2['val_loss'], 'r--', label='Phase 2 Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=len(epochs_phase1), color='gray', linestyle=':', linewidth=2, label='Phase Transition')
    
    # Accuracy plot
    axes[0, 1].plot(epochs_phase1, [a*100 for a in history_phase1['train_acc']], 'b-', label='Phase 1 Train', linewidth=2)
    axes[0, 1].plot(epochs_phase1, [a*100 for a in history_phase1['val_acc']], 'b--', label='Phase 1 Val', linewidth=2)
    axes[0, 1].plot(epochs_phase2_adj, [a*100 for a in history_phase2['train_acc']], 'r-', label='Phase 2 Train', linewidth=2)
    axes[0, 1].plot(epochs_phase2_adj, [a*100 for a in history_phase2['val_acc']], 'r--', label='Phase 2 Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=len(epochs_phase1), color='gray', linestyle=':', linewidth=2)
    axes[0, 1].set_ylim([0, 100])
    
    # Learning rate plot
    all_lrs = history_phase1['learning_rate'] + history_phase2['learning_rate']
    all_epochs = list(epochs_phase1) + epochs_phase2_adj
    axes[1, 0].plot(all_epochs, all_lrs, 'g-', linewidth=2, marker='o')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=len(epochs_phase1), color='gray', linestyle=':', linewidth=2)
    
    # Phase comparison
    phase1_best = max(history_phase1['val_acc'])
    phase2_best = max(history_phase2['val_acc'])
    phases = ['Phase 1\n(Frozen)', 'Phase 2\n(Fine-tuned)']
    accuracies = [phase1_best*100, phase2_best*100]
    colors = ['lightblue', 'lightgreen']
    
    bars = axes[1, 1].bar(phases, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    axes[1, 1].set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Phase Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'complete_training_history.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Training history saved: {os.path.join(config.OUTPUT_DIR, 'complete_training_history.png')}")
    plt.show()


def save_complete_results(results_phase1, results_phase2, config):
    """Save complete training results"""
    results = {
        'training_date': datetime.now().isoformat(),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs_phase1': config.NUM_EPOCHS_PHASE1,
            'epochs_phase2': config.NUM_EPOCHS_PHASE2,
            'learning_rate_phase1': config.LEARNING_RATE_PHASE1,
            'learning_rate_phase2': config.LEARNING_RATE_PHASE2,
            'weight_decay': config.WEIGHT_DECAY,
            'dropout_rate': config.DROPOUT_RATE,
            'optimizer': 'Adam',
            'beta1': config.BETA1,
            'beta2': config.BETA2,
            'device': str(config.DEVICE)
        },
        'phase1_results': {
            'best_val_accuracy': float(results_phase1['best_val_acc']),
            'test_accuracy': float(results_phase1['test_acc']),
            'test_loss': float(results_phase1['test_loss']),
            'precision': float(results_phase1['precision']),
            'recall': float(results_phase1['recall']),
            'f1_score': float(results_phase1['f1_score'])
        },
        'phase2_results': {
            'best_val_accuracy': float(results_phase2['best_val_acc']),
            'test_accuracy': float(results_phase2['test_acc']),
            'test_loss': float(results_phase2['test_loss']),
            'precision': float(results_phase2['precision']),
            'recall': float(results_phase2['recall']),
            'f1_score': float(results_phase2['f1_score'])
        },
        'improvement': {
            'val_accuracy_improvement': float(results_phase2['best_val_acc'] - results_phase1['best_val_acc']),
            'test_accuracy_improvement': float(results_phase2['test_acc'] - results_phase1['test_acc'])
        }
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, 'complete_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Complete results saved: {results_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPLETE TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nPhase 1 (Frozen Layers):")
    print(f"  Best Val Accuracy: {results_phase1['best_val_acc']:.4f} ({results_phase1['best_val_acc']*100:.2f}%)")
    print(f"  Test Accuracy: {results_phase1['test_acc']:.4f} ({results_phase1['test_acc']*100:.2f}%)")
    print(f"  Precision: {results_phase1['precision']:.4f}")
    print(f"  Recall: {results_phase1['recall']:.4f}")
    print(f"  F1-Score: {results_phase1['f1_score']:.4f}")
    
    print(f"\nPhase 2 (Fine-tuned):")
    print(f"  Best Val Accuracy: {results_phase2['best_val_acc']:.4f} ({results_phase2['best_val_acc']*100:.2f}%)")
    print(f"  Test Accuracy: {results_phase2['test_acc']:.4f} ({results_phase2['test_acc']*100:.2f}%)")
    print(f"  Precision: {results_phase2['precision']:.4f}")
    print(f"  Recall: {results_phase2['recall']:.4f}")
    print(f"  F1-Score: {results_phase2['f1_score']:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Val Accuracy: +{results['improvement']['val_accuracy_improvement']*100:.2f}%")
    print(f"  Test Accuracy: +{results['improvement']['test_accuracy_improvement']*100:.2f}%")
    
    print(f"\n{'='*80}")
    
    return results


def main():
    """Main training function"""
    config = TrainingConfig()
    
    print("="*80)
    print("COMPLETE BONE FRACTURE DETECTION TRAINING")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET_ROOT}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    # Normalize dataset path (convert Windows backslashes to forward slashes for Linux/Colab)
    dataset_root_normalized = str(config.DATASET_ROOT).replace('\\', '/')
    # Remove double slashes
    dataset_root_normalized = dataset_root_normalized.replace('//', '/')
    
    # If it's a relative path, make it absolute based on current working directory
    if not os.path.isabs(dataset_root_normalized):
        current_dir = os.getcwd()
        dataset_root_normalized = os.path.join(current_dir, dataset_root_normalized).replace('\\', '/')
    
    print(f"Dataset path (normalized): {dataset_root_normalized}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Path exists: {os.path.exists(dataset_root_normalized)}")
    
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        root_dir=dataset_root_normalized,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        classification_mode=True
    )
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Create model
    print("\nCreating model...")
    model = OptimizedBoneFractureClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        freeze_layers=True
    )
    model = model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Architecture:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen Parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # Phase 1: Training with frozen layers
    results_phase1 = train_model_complete(
        model, train_loader, val_loader, test_loader, config, phase_name='Phase 1'
    )
    
    # Phase 2: Fine-tuning with progressive strategy
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING WITH DIFFERENTIAL LEARNING RATES")
    print("="*80)
    model.unfreeze_all()
    
    trainable_params_phase2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params, classifier_params = model.get_parameter_groups()
    backbone_param_count = sum(p.numel() for p in backbone_params)
    classifier_param_count = sum(p.numel() for p in classifier_params)
    
    print(f"Trainable Parameters (Phase 2): {trainable_params_phase2:,} (100%)")
    print(f"  Backbone Parameters: {backbone_param_count:,} (LR: {config.LEARNING_RATE_BACKBONE:.6f})")
    print(f"  Classifier Parameters: {classifier_param_count:,} (LR: {config.LEARNING_RATE_PHASE2:.6f})")
    print(f"\nFine-tuning Strategy:")
    print(f"  - All layers unfrozen")
    print(f"  - Differential learning rates for stable fine-tuning")
    print(f"  - Backbone: Lower LR ({config.LEARNING_RATE_BACKBONE:.6f}) to preserve pretrained features")
    print(f"  - Classifier: Higher LR ({config.LEARNING_RATE_PHASE2:.6f}) for task-specific adaptation")
    
    results_phase2 = train_model_complete(
        model, train_loader, val_loader, test_loader, config, phase_name='Phase 2'
    )
    
    # Plot training history
    plot_training_history(results_phase1['history'], results_phase2['history'], config)
    
    # Save complete results
    complete_results = save_complete_results(results_phase1, results_phase2, config)
    
    # Save final model
    final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model_complete.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'phase1_results': results_phase1,
        'phase2_results': results_phase2,
        'complete_results': complete_results
    }, final_model_path)
    
    print(f"\n✅ Final model saved: {final_model_path}")
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Accuracy: {results_phase2['test_acc']:.4f} ({results_phase2['test_acc']*100:.2f}%)")
    print(f"All results saved in: {config.OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()

