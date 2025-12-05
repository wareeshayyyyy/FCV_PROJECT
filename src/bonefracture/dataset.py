import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class BoneXrayDataset(Dataset):
    """Simple Dataset expecting a CSV with `image_path,label` columns.

    - `image_path` may be absolute or relative to `root_dir`.
    - `label` should be integer class ids (0..C-1).
    """

    def __init__(self, csv_file, root_dir=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if self.root_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label
