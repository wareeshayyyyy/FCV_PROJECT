import os
import random
from typing import Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import BoneXrayDataset


def calculate_class_weights(dataset) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.

    Returns a torch.Tensor suitable to pass to `nn.CrossEntropyLoss(weight=...)`.
    """
    labels = []
    for _, l in dataset:
        labels.append(int(l))
    counts = pd.Series(labels).value_counts().sort_index()
    # ensure contiguous classes starting at 0
    counts_list = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0).tolist()
    total = sum(counts_list)
    weights = [total / (len(counts_list) * c) if c > 0 else 0.0 for c in counts_list]
    return torch.tensor(weights, dtype=torch.float)


def create_dataloaders(root_dir: str,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       classification_mode: bool = True,
                       val_size: float = 0.1,
                       test_size: float = 0.1,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, BoneXrayDataset, BoneXrayDataset, BoneXrayDataset]:
    """Create train/val/test dataloaders.

    Looks for `train.csv`, `val.csv`, `test.csv` in `root_dir`. If not found,
    looks for `labels.csv` and performs a stratified split.
    CSV format expected: `image_path,label` where `image_path` is relative to `root_dir` or absolute.
    """
    train_csv = os.path.join(root_dir, 'train.csv')
    val_csv = os.path.join(root_dir, 'val.csv')
    test_csv = os.path.join(root_dir, 'test.csv')
    labels_csv = os.path.join(root_dir, 'labels.csv')

    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        train_ds = BoneXrayDataset(train_csv, root_dir=root_dir, transform=default_transform)
        val_ds = BoneXrayDataset(val_csv, root_dir=root_dir, transform=val_transform)
        test_ds = BoneXrayDataset(test_csv, root_dir=root_dir, transform=val_transform)
    elif os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        # stratified split into train / temp, then temp -> val/test
        train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), stratify=df['label'], random_state=random_state)
        rel_val = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, test_size=(1 - rel_val), stratify=temp_df['label'], random_state=random_state)

        tmp_dir = os.path.join(root_dir, 'tmp_splits')
        os.makedirs(tmp_dir, exist_ok=True)
        train_path = os.path.join(tmp_dir, 'train_split.csv')
        val_path = os.path.join(tmp_dir, 'val_split.csv')
        test_path = os.path.join(tmp_dir, 'test_split.csv')
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        train_ds = BoneXrayDataset(train_path, root_dir=root_dir, transform=default_transform)
        val_ds = BoneXrayDataset(val_path, root_dir=root_dir, transform=val_transform)
        test_ds = BoneXrayDataset(test_path, root_dir=root_dir, transform=val_transform)
    else:
        raise FileNotFoundError('No dataset CSVs found in root_dir. Place `labels.csv` or `train.csv`/`val.csv`/`test.csv`.')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
