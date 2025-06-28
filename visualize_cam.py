from cam.gradcam import random_finercam_grid
from models.model import load_model
from utils.data import get_dataloaders, get_class_names
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path
import torch


project_dir = Path(__file__).resolve().parent
weights_path = project_dir / 'models' / 'checkpoints' / 'best_resnet50.pth'
data_dir     = project_dir / 'data'   / 'images'
loaders_dir  = project_dir / 'data'   / 'dataloaders' / 'dataloaders.pth'
output_dir   = project_dir / 'results' / 'finercam_grid'
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Test dataloader
dataloaders = get_dataloaders(loaders_path=str(loaders_dir))
test_loader = dataloaders['test']

# Dataset & classes
full_dataset = ImageFolder(root=str(data_dir), transform=None)
class_names = get_class_names(data_dir)
num_classes = len(class_names)

# Model and target layer
model = load_model(num_classes=num_classes, feature_extract=False)
checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
target_layer = model.layer4[-1]

# FinerCAM grid
random_finercam_grid(
    model=model,
    loader=test_loader,
    full_dataset=full_dataset,
    target_layer=target_layer,
    output_root=str(output_dir),
    device=device,
    num_samples=1,                      # pick 10 random test images
    input_size=224,
    alphas=[0.5, 1.0, 2.0],
    comparison_settings=[[], [1,2]],     # pure CAM and two-way comparison
    seed=15
)
