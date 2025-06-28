import os
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import FinerCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget
from torchvision import transforms
from pathlib import Path
import torch
import random


def random_finercam_grid(model: torch.nn.Module,
                         loader: torch.utils.data.DataLoader,
                         full_dataset: ImageFolder,
                         target_layer: torch.nn.Module,
                         output_root: str,
                         device: torch.device,
                         num_samples: int = 10,
                         input_size: int = 224,
                         alphas: list = [0.5, 1.0, 2.0],
                         comparison_settings: list = [[], [1,2]], # If empty, no comparison so classic GradCAM
                         seed: int = 15):
    """
    For `num_samples` random test images, run FinerCAM with each combination
    of comparison_categories and alpha, and save overlays in per-image folders.

    Args:
        model:           your fine-tuned model
        loader:          DataLoader for the test set
        full_dataset:    the ImageFolder used to build the loader (no transforms)
        target_layer:    the single nn.Module (e.g. model.layer4)
        output_root:     root directory for output (will mkdir)
        device:          torch.device
        num_samples:     number of random test images to visualize
        input_size:      height/width to resize
        alphas:          list of alpha weights to try
        comparison_settings: list of comparison_categories lists to try
        seed:            random seed for reproducibility

    Explanation of special parameters:
        FinerCAM is a wrapper around GradCAM that allows for comparison
        between multiple categories by computing the difference in activations
        between the target category and the comparison categories.
        The 'target_layer' is the layer from which to extract the feature maps
        The `comparison_categories` argument specifies which categories to compare
        against the target category. If empty, it behaves like GradCAM.
        The `alpha` parameter controls the weight of the comparison categories
        in the final mask. A value of 1.0 means equal weight, while
        values less than 1.0 reduce their influence, and values greater than 1.0 increase it.
    """

    #  Preprocessing (for model) and visualization transforms
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # Grab the test set indices from the loader
    test_subset: Subset = loader.dataset
    test_indices = test_subset.indices

    # Randomly choose `num_samples` indices from the test set
    random.seed(seed)
    chosen = random.sample(test_indices, min(num_samples, len(test_indices)))

    # Instantiate FinerCAM (wrapping GradCAM)
    model.to(device).eval()
    cam = FinerCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=None,
        base_method=GradCAM
    )

    # Loop over the images
    for idx in chosen:
        img_path, true_label = full_dataset.samples[idx]
        img_name = Path(img_path).stem
        out_dir = Path(output_root) / img_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # load and prepare raw image for viz
        pil = Image.open(img_path).convert("RGB")
        pil_resized = pil.resize((input_size, input_size))
        img_np = np.array(pil_resized, dtype=np.float32) / 255.0

        # prepare tensor for model
        inp = preprocess(pil).unsqueeze(0).to(device)

        # top-1 prediction
        with torch.no_grad():
            logits = model(inp)
            pred = logits.argmax(dim=1).item()

        targets = [
    FinerWeightedTarget(primary, comparison_categories=comp, alpha=alpha)
]
        # Run FinerCAM for each combination of comparison categories and alpha
        for comp in comparison_settings:
            for alpha in alphas:
                mask = cam(
                    inp,
                    targets=None,
                    eigen_smooth=True,
                    alpha=alpha,
                    comparison_categories=comp,
                    target_idx=pred
                )
                mask = np.squeeze(mask)
                mask = np.clip(mask, 0, 1)

                overlay = show_cam_on_image(img_np, mask, use_rgb=True)
                comp_str = "none" if not comp else "_".join(map(str, comp))
                fname = f"{img_name}_comp-{comp_str}_a-{alpha}.png"
                Image.fromarray(overlay).save(out_dir / fname)

        print(f"Saved {len(alphas)*len(comparison_settings)} overlays for {img_name} in {out_dir}")