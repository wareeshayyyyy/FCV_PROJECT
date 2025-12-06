import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


# ==================== DATASET STRUCTURE ====================
"""
Expected dataset structure (example):
root/
  train/images/*.jpg
  train/labels/*.txt  (YOLO format)
  valid/images
  valid/labels
  test/images
  test/labels

Empty .txt files indicate "Normal" (no fracture) images
"""


class BoneFractureDatasetYOLO(Dataset):
    """Custom Dataset for Bone Fracture Detection (YOLO format)

    Adapted for classification tasks from an object detection dataset.
    """

    YOLO_CLASS_NAMES = {
        0: 'Elbow Positive',
        1: 'Fingers Positive',
        2: 'Forearm Fracture',
        4: 'Humerus Fracture',
        5: 'Shoulder Fracture',
        6: 'Wrist Positive'
    }

    CLASSIFICATION_MAP = {
        'normal': 0,
        'fractured': 1,
        'disease': 2
    }

    def __init__(self, root_dir, split='train', transform=None,
                 classification_mode=True, include_bbox=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classification_mode = classification_mode
        self.include_bbox = include_bbox

        # Normalize path separators (handle both Windows and Linux paths)
        # Convert to string and replace all backslashes with forward slashes
        root_dir = str(root_dir).replace('\\', '/')
        # Remove any double slashes
        root_dir = root_dir.replace('//', '/')
        
        # Use forward slashes for path joining (works on both Windows and Linux)
        self.images_dir = f"{root_dir}/{split}/images"
        self.labels_dir = f"{root_dir}/{split}/labels"
        
        # Normalize paths (remove any double slashes, etc.) and ensure forward slashes
        self.images_dir = self.images_dir.replace('//', '/').replace('\\', '/')
        self.labels_dir = self.labels_dir.replace('//', '/').replace('\\', '/')
        
        # Store normalized root_dir
        self.root_dir = root_dir

        # Debug: Print the path being used
        print(f"DEBUG: Checking images directory: {self.images_dir}")
        
        if not os.path.exists(self.images_dir):
            # Try alternative path formats
            alt_paths = [
                self.images_dir,
                self.images_dir.replace('/', '\\'),
                os.path.normpath(self.images_dir),
            ]
            error_msg = f"Images directory not found: {self.images_dir}\n"
            error_msg += f"Tried paths: {alt_paths}\n"
            error_msg += f"Please check:\n"
            error_msg += f"  1. Dataset is uploaded correctly\n"
            error_msg += f"  2. Dataset structure: {root_dir}/{split}/images/\n"
            error_msg += f"  3. Current working directory: {os.getcwd()}\n"
            error_msg += f"  4. Path is correct (use forward slashes in Colab)"
            raise FileNotFoundError(error_msg)

        self.image_files = sorted([f for f in os.listdir(self.images_dir)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        self.data = self._parse_dataset()

        print(f"\n{split.upper()} Dataset loaded:")
        print(f"  Total images: {len(self.data)}")
        self._print_class_distribution()

    def _parse_dataset(self):
        data = []
        for img_file in self.image_files:
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            img_path = os.path.join(self.images_dir, img_file)

            has_fracture = False
            fracture_classes = []
            bboxes = []

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 0:
                    has_fracture = True
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            fracture_classes.append(class_id)
                            if self.include_bbox:
                                bbox = [float(x) for x in parts[1:5]]
                                bboxes.append(bbox)

            if self.classification_mode:
                label = 1 if has_fracture else 0
            else:
                if has_fracture:
                    label = fracture_classes[0] + 1 if len(fracture_classes) > 0 else 1
                else:
                    label = 0

            data.append({
                'image_path': img_path,
                'label': label,
                'has_fracture': has_fracture,
                'fracture_classes': fracture_classes,
                'bboxes': bboxes
            })

        return data

    def _print_class_distribution(self):
        labels = [item['label'] for item in self.data]
        counter = Counter(labels)
        if self.classification_mode:
            class_names = ['Normal', 'Fractured', 'Disease']
        else:
            class_names = ['Normal'] + [self.YOLO_CLASS_NAMES.get(i, f'Class {i}') for i in range(7)]

        print("\n  Class Distribution:")
        for label, count in sorted(counter.items()):
            class_name = class_names[label] if label < len(class_names) else f'Unknown ({label})'
            percentage = (count / len(self.data)) * 100 if len(self.data) > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        if self.include_bbox and len(item['bboxes']) > 0:
            return image, label, torch.tensor(item['bboxes'])
        else:
            return image, label

    def visualize_sample(self, idx, show_bbox=True):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        if show_bbox and len(item['bboxes']) > 0:
            img_width, img_height = image.size
            for bbox, cls in zip(item['bboxes'], item['fracture_classes']):
                x_center, y_center, width, height = bbox
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                class_name = self.YOLO_CLASS_NAMES.get(cls, f'Class {cls}')
                ax.text(x1, y1 - 10, class_name, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        label_name = 'Fractured' if item['has_fracture'] else 'Normal'
        ax.set_title(f"Label: {label_name}\n{item['image_path']}", fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


def create_dataloaders(root_dir, batch_size=32, num_workers=4, classification_mode=True):
    # Normalize path (convert Windows backslashes to forward slashes)
    root_dir = str(root_dir).replace('\\', '/')
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BoneFractureDatasetYOLO(root_dir=root_dir, split='train', transform=train_transform, classification_mode=classification_mode)
    val_dataset = BoneFractureDatasetYOLO(root_dir=root_dir, split='valid', transform=val_test_transform, classification_mode=classification_mode)
    test_dataset = BoneFractureDatasetYOLO(root_dir=root_dir, split='test', transform=val_test_transform, classification_mode=classification_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def calculate_class_weights(dataset):
    labels = [item['label'] for item in dataset.data]
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = max(class_counts.keys()) + 1 if len(class_counts) > 0 else 1
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total / (num_classes * class_counts[i])
        else:
            weight = 0.0
        weights.append(weight)
    return torch.FloatTensor(weights)


def main():
    DATASET_ROOT = os.getenv('DATASET_ROOT', '.')
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(root_dir=DATASET_ROOT, batch_size=32, num_workers=4, classification_mode=True)
    class_weights = calculate_class_weights(train_dataset)
    print(f"\nCalculated class weights: {class_weights}")
    for idx in range(min(5, len(train_dataset))):
        train_dataset.visualize_sample(idx, show_bbox=True)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels in batch: {labels}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Total training images: {len(train_dataset)}")


if __name__ == '__main__':
    main()
