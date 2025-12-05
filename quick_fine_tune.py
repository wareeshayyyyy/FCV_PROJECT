"""
QUICK FINE-TUNING SCRIPT - Optimized for fast training
Fine-tunes DenseNet-121 with optimized hyperparameters
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append('src')
from bonefracture.bone_yolo_dataset import create_dataloaders, calculate_class_weights


class QuickFineTuneConfig:
    """Quick training config for fast results"""
    DATASET_ROOT = r'data\archive\bone fracture detection.v4-v4.yolov8'
    CHECKPOINT_DIR = './checkpoints'
    OUTPUT_DIR = './training_results'
    
    NUM_CLASSES = 2
    PRETRAINED = True
    BATCH_SIZE = 32  # Larger batch for faster training
    NUM_EPOCHS = 5   # Quick training
    
    # Optimized learning rates
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    
    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


class FineTunedBoneFractureClassifier(nn.Module):
    """DenseNet-121 with fine-tuning support"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(FineTunedBoneFractureClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Replace classifier
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
        
        # Initialize classifier
        for m in self.densenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.densenet(x)
    
    def freeze_backbone(self):
        """Freeze backbone, train only classifier"""
        for param in self.densenet.features.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True


def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def main():
    config = QuickFineTuneConfig()
    
    print("="*80)
    print("QUICK FINE-TUNING - DenseNet-121")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("="*80)
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        root_dir=config.DATASET_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        classification_mode=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create model
    print("\nCreating model...")
    model = FineTunedBoneFractureClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    )
    model = model.to(config.DEVICE)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    
    # Phase 1: Train classifier only
    print("\n" + "="*80)
    print("PHASE 1: Training Classifier (Backbone Frozen)")
    print("="*80)
    model.freeze_backbone()
    
    class_weights = calculate_class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    best_val_acc = 0.0
    history_phase1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        history_phase1['train_loss'].append(train_loss)
        history_phase1['train_acc'].append(train_acc)
        history_phase1['val_loss'].append(val_loss)
        history_phase1['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config.__dict__
            }, os.path.join(config.CHECKPOINT_DIR, 'best_phase1.pth'))
            print("✅ Best model saved!")
    
    # Phase 2: Fine-tune all layers
    print("\n" + "="*80)
    print("PHASE 2: Fine-tuning All Layers")
    print("="*80)
    model.unfreeze_all()
    
    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE * 0.1,  # 10x lower
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    history_phase2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        history_phase2['train_loss'].append(train_loss)
        history_phase2['train_acc'].append(train_acc)
        history_phase2['val_loss'].append(val_loss)
        history_phase2['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config.__dict__
            }, os.path.join(config.CHECKPOINT_DIR, 'best_fine_tuned.pth'))
            print("✅ Best model saved!")
    
    # Final test evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    test_loss, test_acc = validate(model, test_loader, criterion, config.DEVICE)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    # Get predictions
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.NUM_EPOCHS,
            'learning_rate_phase1': config.LEARNING_RATE,
            'learning_rate_phase2': config.LEARNING_RATE * 0.1,
            'weight_decay': config.WEIGHT_DECAY,
            'device': str(config.DEVICE)
        },
        'phase1_best_val_acc': float(max(history_phase1['val_acc'])),
        'phase2_best_val_acc': float(max(history_phase2['val_acc'])),
        'test_metrics': {
            'accuracy': float(test_acc),
            'loss': float(test_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, 'fine_tuning_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINE-TUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nPhase 1 (Frozen Backbone):")
    print(f"  Best Val Accuracy: {max(history_phase1['val_acc']):.4f} ({max(history_phase1['val_acc'])*100:.2f}%)")
    print(f"\nPhase 2 (Fine-tuned):")
    print(f"  Best Val Accuracy: {max(history_phase2['val_acc']):.4f} ({max(history_phase2['val_acc'])*100:.2f}%)")
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\n✅ Results saved: {results_path}")
    print(f"✅ Model saved: {os.path.join(config.CHECKPOINT_DIR, 'best_fine_tuned.pth')}")
    print("="*80)


if __name__ == '__main__':
    main()

