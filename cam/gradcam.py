import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image_paths: list[str],
    output_dir: str,
    device: torch.device,
    input_size: int = 224,
):
    """
    Apply vanilla GradCAM to each image and save overlays.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device).eval()

    # Transform to apply to input images
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # Initialize GradCAM
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    for img_path in image_paths:
        # load + preprocess
        pil = Image.open(img_path).convert("RGB")
        pil_resized = pil.resize((input_size, input_size))
        img_np = np.array(pil_resized, dtype=np.float32) / 255.0
        inp = preprocess(pil).unsqueeze(0).to(device)

        # predict
        with torch.no_grad():
            logits = model(inp)
            pred = logits.argmax(dim=1).item()

        # run GradCAM
        targets = [ClassifierOutputTarget(pred)]
        mask = cam(input_tensor=inp, targets=targets)[0]
        mask = np.clip(mask, 0, 1)

        overlay = show_cam_on_image(img_np, mask, use_rgb=True)
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{name}_gradcam.png")
        Image.fromarray(overlay).save(out_path)