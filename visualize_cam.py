from cam.finercam import finercam
from cam.gradcam import gradcam
from models.model import load_model
from utils.data import get_num_classes, sample_random_images_by_class, sample_images_from_class
from pathlib import Path
import torch



project_dir = Path(__file__).resolve().parent
weights_path = project_dir / 'models' / 'checkpoints' / 'best_resnet50.pth'
data_dir     = project_dir / 'data' 
images_dir     = data_dir   / 'images'
loaders_dir  = project_dir / 'data'   / 'dataloaders' / 'dataloaders.pth'
output_dir_fc   = project_dir / 'results' / 'finerCAM'
output_dir_gc   = project_dir / 'results' / 'gradCAM'
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = 'data/images/n02088466-bloodhound/n02088466_1262.jpg'
num_classes = get_num_classes(images_dir)


# Model and target layer
model = load_model(num_classes=num_classes, feature_extract=False)
checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
target_layer = model.layer4[-1]






if __name__ == "__main__":
    
    alpha_ranges = [0.2, 0.5, 1.0]
    comparison_categories = [[1], [1,2]]

    from pathlib import Path
    images_dir = Path(images_dir)

    # For random images
    # images_paths = sample_random_images_by_class(data_dir = images_dir,
    #                                             num_samples = 20,
    #                                             exts = ".jpg")
    

    images_path = sample_images_from_class(data_dir=images_dir,
                                           class_name = 'n02085620-Chihuahua',
                                           num_samples=20,
                                           exts=".jpg")

    for alpha in alpha_ranges:
        for comparison in comparison_categories:
            output_dir_fc_r = output_dir_fc / f'alpha_{alpha}_comparison_{comparison}'
            output_dir_fc_r.mkdir(parents=True, exist_ok=True)

            finercam(
                model=model,
                target_layer=target_layer,
                image_paths= images_path, 
                output_dir=str(output_dir_fc_r),
                device=device,
                input_size=224,
                alpha= alpha,
                comparison_categories= comparison)

    gradcam(
        model=model,
        target_layer=target_layer,
        image_paths=images_path,  
        output_dir=str(output_dir_gc),
        device=device,
        input_size=224
    )