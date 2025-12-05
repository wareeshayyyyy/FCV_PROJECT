"""Full training pipeline for Bone Fracture Detection (DenseNet121).

This module implements the training script you provided, adapted to use the
`create_dataloaders` and `calculate_class_weights` helpers in
`src/bonefracture/bone_dataset.py`.

To run:
    set PYTHONPATH to the project's `src` directory, then:
    python -m bonefracture.train_full

Example (PowerShell):
    $env:PYTHONPATH = 'C:/bone_fracture_densenet/src'; python -m bonefracture.train_full
"""
import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .bone_dataset import create_dataloaders, calculate_class_weights


# ==================== CONFIGURATION ====================
class Config:
    """Training configuration"""
    DATASET_ROOT = os.getenv('DATASET_ROOT', '.')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './outputs')
    CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
    MODEL_NAME = 'densenet121'
    NUM_CLASSES = 3
    PRETRAINED = True
    FREEZE_LAYERS = True
    BATCH_SIZE = 16
    NUM_EPOCHS_PHASE1 = 2
    NUM_EPOCHS_PHASE2 = 2
    LEARNING_RATE_PHASE1 = 1e-4
    LEARNING_RATE_PHASE2 = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATIENCE = 3
    MIN_DELTA = 0.001
    SAVE_BEST_ONLY = True
    SAVE_EVERY_N_EPOCHS = 5

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        self.save_config()

    def save_config(self):
        config_dict = {k: str(v) for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(os.path.join(self.OUTPUT_DIR, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)


# ==================== MODEL DEFINITION ====================
class BoneFractureClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_layers=True):
        super(BoneFractureClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)

        if freeze_layers:
            for param in self.densenet.features.parameters():
                param.requires_grad = False
            # unfreeze last block and norm
            try:
                for param in self.densenet.features.denseblock4.parameters():
                    param.requires_grad = True
                for param in self.densenet.features.norm5.parameters():
                    param.requires_grad = True
            except Exception:
                pass

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

    def forward(self, x):
        return self.densenet(x)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# ==================== TRAINING UTILITIES ====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)

    def plot(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].legend()
        axes[0, 1].plot(self.train_accs, label='Train Acc')
        axes[0, 1].plot(self.val_accs, label='Val Acc')
        axes[0, 1].legend()
        axes[1, 0].plot(self.learning_rates)
        loss_diff = [t - v for t, v in zip(self.train_losses, self.val_losses)]
        axes[1, 1].plot(loss_diff)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# ==================== TRAIN/VALID ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    return epoch_loss, epoch_acc.item()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, phase_name='Phase 1'):
    best_acc = 0.0
    best_model_wts = None
    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    num_epochs = config.NUM_EPOCHS_PHASE1 if phase_name == 'Phase 1' else config.NUM_EPOCHS_PHASE2
    print(f"{phase_name}: Training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        current_lr = optimizer.param_groups[0]['lr']
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr)
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict().copy()
            save_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_{phase_name.replace(" ", "_").lower()}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc, 'val_loss': val_loss}, save_path)
            print(f'  ✓ New best model saved!')
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            save_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    plot_path = os.path.join(config.OUTPUT_DIR, f'training_history_{phase_name.replace(" ", "_")}.png')
    metrics.plot(plot_path)
    return model, metrics


# ==================== EVALUATION ====================
def evaluate_model(model, test_loader, config, class_names=None):
    if class_names is None:
        class_names = ['Normal', 'Fractured', 'Disease']
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    with open(os.path.join(config.OUTPUT_DIR, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        print(f'Weighted ROC-AUC Score: {roc_auc:.4f}')
    except Exception:
        print('Could not calculate ROC-AUC (need multiple classes in test set)')
    return all_preds, all_labels, all_probs


def main():
    config = Config()
    print(f'Using device: {config.DEVICE}')
    print('Loading dataset...')
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(root_dir=config.DATASET_ROOT, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    class_weights = calculate_class_weights(train_ds)
    print(f'Class weights: {class_weights}')
    model = BoneFractureClassifier(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED, freeze_layers=config.FREEZE_LAYERS)
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    optimizer = optim.Adam([
        {'params': model.densenet.features.parameters(), 'lr': config.LEARNING_RATE_PHASE1 * 0.1},
        {'params': model.densenet.classifier.parameters(), 'lr': config.LEARNING_RATE_PHASE1}
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    model, metrics_phase1 = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, phase_name='Phase 1')
    print('Unfreezing all layers for fine-tuning')
    model.unfreeze_all()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_PHASE2, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    model, metrics_phase2 = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, phase_name='Phase 2')
    print('Final evaluation on test set')
    predictions, labels, probabilities = evaluate_model(model, test_loader, config)
    final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    torch.save({'model_state_dict': model.state_dict(), 'config': config.__dict__}, final_model_path)
    print('Training complete')


if __name__ == '__main__':
    main()
