import torch
import torch.nn as nn
from torchvision import models


class BoneFractureClassifier(nn.Module):
    """DenseNet121 fine-tuned for bone fracture detection.

    Args:
        num_classes (int): number of output classes
        pretrained (bool): load ImageNet pretrained weights
    """

    def __init__(self, num_classes=3, pretrained=True):
        super(BoneFractureClassifier, self).__init__()
        # Load torchvision DenseNet121
        # Use the legacy `pretrained` argument for compatibility
        self.densenet = models.densenet121(pretrained=pretrained)

        # Replace classifier head
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
