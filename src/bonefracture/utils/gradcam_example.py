"""Minimal Grad-CAM example using `grad-cam` package."""
from PIL import Image
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def run_gradcam(model, img_path, target_layer, target_category=None, device='cuda'):
    model.eval()
    model.to(device)

    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    cam = GradCAM(model=model.densenet, target_layers=[target_layer], use_cuda=(device=='cuda'))
    targets = None
    if target_category is not None:
        targets = [ClassifierOutputTarget(target_category)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # convert image to numpy for overlay
    import numpy as np
    img_np = np.array(img.resize((224, 224))).astype(float) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization
