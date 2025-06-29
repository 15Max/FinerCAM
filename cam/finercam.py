import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import FinerCAM


def finercam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image_paths: list,
    output_dir: str,
    device: torch.device,
    input_size: int = 224,
    alpha: float = 1.0,
    comparison_categories: list = [1, 2, 3],
    target_idx: int = None
):
    """
    Run FinerCAM on a list of images and save overlays.

    Args:
        model:            a fine‐tuned model (already loaded & on CPU/GPU)
        target_layer:     the nn.Module to attach FinerCAM to (e.g. model.layer4)
        image_paths:      list of filepaths to input images
        output_dir:       directory where to save cam overlays
        device:           torch.device("cuda") or ("cpu")
        input_size:       int, image resize size (default 224)
        alpha:            float, weight for the primary target in FinerWeightedTarget
        comparison_categories: list of int, other class indices to compare against
        target_idx:       int or None, if provided use this class index instead of auto‐select
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.eval()

    # Transform to apply to input images
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Finercam
    cam = FinerCAM(
        model=model,
        target_layers= [target_layer],
        reshape_transform=None
    )

    for img_path in image_paths:
        # load and prep
        rgb = Image.open(img_path).convert("RGB")
        rgb_resized = rgb.resize((input_size, input_size))
        input_tensor = preprocess(rgb).unsqueeze(0).to(device)

        # decide primary target (default to most similar class)
        with torch.no_grad():
            logits = model(input_tensor)
            pred = logits.argmax(dim=1).item()
        primary_idx = pred if target_idx is None else target_idx

        # run FinerCAM
        mask = cam(
            input_tensor,
            targets=None,               # auto‐select or override below
            eigen_smooth=True,
            alpha=alpha,
            comparison_categories=comparison_categories,
            target_idx=primary_idx
        )
        # mask is H×W float np array in [0,1]

        # overlay
        img_np = np.array(rgb_resized, dtype=np.float32) / 255.0
        # mask might be (1, H, W) or float32; convert to (H, W) uint8-friendly
        mask = np.squeeze(mask)             # drop any singleton dims → (H, W)
        mask = np.clip(mask, 0, 1)          # ensure in [0,1]
        # now show_cam will multiply by 255 and cas
        overlay = show_cam_on_image(img_np, mask, use_rgb=True)

        # save
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{name}_finer_cam.png")
        Image.fromarray(overlay).save(out_path)

