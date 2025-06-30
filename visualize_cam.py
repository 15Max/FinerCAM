
from CAM import GradCAM, FinerCAM
from models.model import load_model
from utils.data import get_num_classes, save_class_names_and_indexes 
from utils.graphs import compute_deletion_curve, compute_relative_confidence_drop, interactive_mask_slider_script
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
model.to(device)
model.eval()






if __name__ == "__main__":
    

    # Create the class names and indexes file
    class_names_file = data_dir / 'class_names_and_indexes.csv'
    if not class_names_file.exists():
        save_class_names_and_indexes(data_dir=str(images_dir), save_path=str(class_names_file))


    single_image = str("data/images/n02109961-Eskimo_dog/n02109961_623.jpg")


    
    gradcam = GradCAM(
        model=model,
        target_layer=target_layer)
    
    # What are the most similar classes to the target class?

    outputs = model(gradcam.preprocess_image(image_path=single_image, device=device)[0])
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted class index: {predicted.item()}")

    # Top 3
    topk_values, topk_indices = torch.topk(outputs, k=3)
    print("Top 3 class indices:", topk_indices[0].tolist())
    

    image_tensor, image = gradcam.preprocess_image(image_path = single_image, device = device)

    cam = gradcam.generate_CAM(
        input_image=image_tensor,
        target_class=97)
    
    overlay_image = gradcam.visualize_CAM(cam, image, show = True, alpha = 0.5)


    finercam = FinerCAM(
        model=model,
        target_layer=target_layer)
    
    cam = finercam.generate_CAM(
        input_image=image_tensor,
        target_class=97,
        reference_classes=99,
        gamma=1
    )

    overlay_image = finercam.visualize_CAM(cam, image, show = True, alpha = 0.5)

    results = compute_deletion_curve(
        model=model,
        input_tensor=image_tensor,
        cam=cam,
        target_class=97,
        reference_class=99,
        steps=50
    )


    interactive_mask_slider_script(model, image_tensor, cam, target_class=97)