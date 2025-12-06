"""
SegNet: Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
================================================================================
SegNet implementation for bone fracture segmentation in X-ray images.

SegNet Architecture:
- Encoder: VGG-16 based encoder with max-pooling indices
- Decoder: Symmetric decoder with upsampling using max-pooling indices
- Pixel-wise classification for fracture region segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import sys

# Add src to path
sys.path.append('src')
from bonefracture.bone_yolo_dataset import BoneFractureDatasetYOLO


class SegNet(nn.Module):
    """
    SegNet: Encoder-Decoder Architecture for Semantic Segmentation
    
    Architecture:
    - Encoder: VGG-16 based (13 convolutional layers)
    - Decoder: Symmetric decoder with upsampling
    - Output: Pixel-wise classification (2 classes: Normal, Fracture)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        
        # Encoder (VGG-16 based)
        vgg = models.vgg16_bn(pretrained=pretrained)
        encoder_features = list(vgg.features.children())
        
        # Encoder blocks
        self.encoder1 = nn.Sequential(*encoder_features[0:7])   # Conv + BN + ReLU + MaxPool
        self.encoder2 = nn.Sequential(*encoder_features[7:14])  # Conv + BN + ReLU + MaxPool
        self.encoder3 = nn.Sequential(*encoder_features[14:24]) # Conv + BN + ReLU + MaxPool
        self.encoder4 = nn.Sequential(*encoder_features[24:34]) # Conv + BN + ReLU + MaxPool
        self.encoder5 = nn.Sequential(*encoder_features[34:44]) # Conv + BN + ReLU + MaxPool
        
        # Decoder blocks (symmetric to encoder)
        self.decoder5 = self._make_decoder_layer(512, 512)
        self.decoder4 = self._make_decoder_layer(512, 256)
        self.decoder3 = self._make_decoder_layer(256, 128)
        self.decoder2 = self._make_decoder_layer(128, 64)
        self.decoder1 = self._make_decoder_layer(64, 64)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _make_decoder_layer(self, in_channels, out_channels):
        """Create a decoder layer with upsampling and convolutions"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path (with max-pooling indices)
        x1, indices1 = self._max_pool_with_indices(self.encoder1(x))
        x2, indices2 = self._max_pool_with_indices(self.encoder2(x1))
        x3, indices3 = self._max_pool_with_indices(self.encoder3(x2))
        x4, indices4 = self._max_pool_with_indices(self.encoder4(x3))
        x5, indices5 = self._max_pool_with_indices(self.encoder5(x4))
        
        # Decoder path (upsampling using indices)
        x = self._max_unpool_with_indices(x5, indices5, output_size=x4.size())
        x = self.decoder5(x)
        
        x = self._max_unpool_with_indices(x, indices4, output_size=x3.size())
        x = self.decoder4(x)
        
        x = self._max_unpool_with_indices(x, indices3, output_size=x2.size())
        x = self.decoder3(x)
        
        x = self._max_unpool_with_indices(x, indices2, output_size=x1.size())
        x = self.decoder2(x)
        
        x = self._max_unpool_with_indices(x, indices1, output_size=x.size())
        x = self.decoder1(x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x
    
    def _max_pool_with_indices(self, x):
        """Max pooling that returns both output and indices"""
        # For simplicity, we'll use regular max pooling and track indices separately
        # In practice, SegNet uses MaxUnpool2d which requires indices
        output = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        return output
    
    def _max_unpool_with_indices(self, x, indices, output_size):
        """Upsampling using max-pooling indices"""
        return F.max_unpool2d(x, indices, kernel_size=2, stride=2, output_size=output_size)


class SegNetSimplified(nn.Module):
    """
    Simplified SegNet for bone fracture segmentation
    Uses a simpler architecture suitable for binary segmentation
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(SegNetSimplified, self).__init__()
        self.num_classes = num_classes
        
        # Encoder (VGG-16 based)
        vgg = models.vgg16_bn(pretrained=pretrained)
        encoder_features = list(vgg.features.children())
        
        # Encoder blocks
        self.encoder1 = nn.Sequential(*encoder_features[0:7])
        self.encoder2 = nn.Sequential(*encoder_features[7:14])
        self.encoder3 = nn.Sequential(*encoder_features[14:24])
        self.encoder4 = nn.Sequential(*encoder_features[24:34])
        self.encoder5 = nn.Sequential(*encoder_features[34:44])
        
        # Decoder blocks
        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)  # 224x224 -> 112x112
        e2 = self.encoder2(e1)  # 112x112 -> 56x56
        e3 = self.encoder3(e2)  # 56x56 -> 28x28
        e4 = self.encoder4(e3)  # 28x28 -> 14x14
        e5 = self.encoder5(e4)  # 14x14 -> 7x7
        
        # Decoder path (with skip connections via upsampling)
        d5 = F.interpolate(self.decoder5(e5), size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.decoder4(d5), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.decoder3(d4), size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.decoder2(d3), size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = F.interpolate(self.decoder1(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Final classification
        output = self.final_conv(d1)
        
        return output


def create_segmentation_masks_from_labels(image_path, label_path, output_size=(224, 224)):
    """
    Create segmentation masks from YOLO format labels
    For bone fracture segmentation, we create binary masks
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Read YOLO labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized)
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    width = float(parts[3]) * w
                    height = float(parts[4]) * h
                    
                    # Create bounding box mask (simplified - in practice, use actual segmentation masks)
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Mark fracture region (class_id 0 = normal, 1 = fractured)
                    if class_id == 1:  # Fractured
                        mask[y1:y2, x1:x2] = 1
    
    # Resize mask to output size
    mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized


def visualize_segmentation_results(model, test_loader, device, num_samples=6, save_path='segnet_results.png'):
    """
    Visualize SegNet segmentation results
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            predictions = F.softmax(outputs, dim=1)
            predicted_masks = torch.argmax(predictions, dim=1).cpu().numpy()
            
            # Original image
            img = images[0].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            # Predicted mask
            axes[idx, 1].imshow(predicted_masks[0], cmap='gray')
            axes[idx, 1].set_title('Predicted Segmentation')
            axes[idx, 1].axis('off')
            
            # Overlay
            overlay = img.copy()
            mask_overlay = predicted_masks[0] > 0
            overlay[mask_overlay] = [1, 0, 0]  # Red overlay for fracture regions
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title('Overlay (Red = Fracture)')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Segmentation results saved to: {save_path}")
    plt.close()


def demo_segnet_inference():
    """
    Demo function to show SegNet inference on sample images
    """
    print("="*80)
    print("SEGNET: BONE FRACTURE SEGMENTATION DEMO")
    print("="*80)
    
    # Check for trained model
    model_path = 'checkpoints/segnet_bone_fracture.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SegNetSimplified(num_classes=2, pretrained=True)
    model = model.to(device)
    
    if os.path.exists(model_path):
        print(f"[LOADING] Loading trained SegNet model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("[OK] Model loaded successfully")
    else:
        print("[INFO] No trained model found. Using pretrained encoder only.")
        print("[NOTE] For full segmentation, train the model first.")
        model.eval()
    
    # Load test dataset
    dataset_root = r'data\archive\bone fracture detection.v4-v4.yolov8'
    dataset_root = str(Path(dataset_root).resolve())
    
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Dataset not found at: {dataset_root}")
        return
    
    # Create test dataset with appropriate transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = BoneFractureDatasetYOLO(
        root_dir=dataset_root,
        split='test',
        transform=test_transform,
        classification_mode=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print(f"[INFO] Testing on {len(test_dataset)} test images")
    print("[INFO] Generating segmentation visualizations...")
    
    # Visualize results
    visualize_segmentation_results(
        model, 
        test_loader, 
        device, 
        num_samples=6,
        save_path='complete_results/segnet_segmentation_results.png'
    )
    
    print("\n" + "="*80)
    print("[SUCCESS] SegNet demo complete!")
    print("="*80)
    print("Results saved to: complete_results/segnet_segmentation_results.png")
    print("="*80)


if __name__ == '__main__':
    demo_segnet_inference()

