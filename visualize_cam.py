
from CAM import GradCAM, FinerCAM
from models.model import load_model
from utils.data import get_num_classes, save_class_names_and_indexes 
from utils.graphs import compute_deletion_curve, compute_relative_confidence_drop, interactive_mask_slider_script
from pathlib import Path
import torch
import os



project_dir = Path(__file__).resolve().parent
weights_path = project_dir / 'models' / 'checkpoints' / 'best_resnet50.pth'
data_dir     = project_dir / 'data' 
images_dir     = data_dir   / 'images'
loaders_dir  = project_dir / 'data'   / 'dataloaders' / 'dataloaders.pth'
visualizations_dir = project_dir / 'results' / 'visualizations'
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = get_num_classes(images_dir)

os.makedirs(visualizations_dir, exist_ok=True)

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


    husky_target = str("data/images/n02110185-Siberian_husky/n02110185_2736.jpg")
    eskimo_dog_reference = str("data/images/n02109961-Eskimo_dog/n02109961_1017.jpg")



    gradcam = GradCAM(
        model=model,
        target_layer=target_layer)
    
    # What are the most similar classes to the target class?

    outputs = model(gradcam.preprocess_image(image_path=husky_target, device=device)[0])
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted class index: {predicted.item()}")

    # Top 3
    topk_values, topk_indices = torch.topk(outputs, k=3)
    print("Top 3 class indices:", topk_indices[0].tolist())
    

    image_tensor, image = gradcam.preprocess_image(image_path = husky_target, device = device)

    cam_husky = gradcam.generate_CAM(
        input_image=image_tensor,
        target_class=99)
    
    
    husky_dog_with_husky_fm = gradcam.visualize_CAM(cam_husky, image, show = True, alpha = 0.5, saving_path=str(visualizations_dir / 'husky_dog_with_husky_fm.jpg'))

    cam_eskimo = gradcam.generate_CAM(
        input_image=image_tensor,
        target_class=97)
    
    husky_dog_with_eskimo_fm = gradcam.visualize_CAM(cam_eskimo, image, show = True, alpha = 0.5, saving_path=str(visualizations_dir / 'husky_dog_with_eskimo_fm.jpg'))


    finercam = FinerCAM(
        model=model,
        target_layer=target_layer)
    
    finercam_eskimo = finercam.generate_CAM(
        input_image=image_tensor,
        target_class=99,
        reference_classes=97,
        gamma=1
    )

    husky_dog_finercam_ref_eskimo = finercam.visualize_CAM(finercam_eskimo, image, show = True, alpha = 0.5, saving_path=str(visualizations_dir / 'husky_dog_finercam_ref_eskimo.jpg'))

    results = compute_deletion_curve(
        model=model,
        input_tensor=image_tensor,
        cam=finercam_eskimo,
        target_class=99,
        reference_class=97,
        steps=100,
    )


    interactive_mask_slider_script(model, image_tensor, finercam_eskimo, target_class=97)

     
    # Testing several gamma values for FinerCAM
    gammas = [0.2, 0.4, 0.6, 0.8, 1.0]
    for gamma in gammas:
        finercam_eskimo = finercam.generate_CAM(
            input_image=image_tensor,
            target_class=99,
            reference_classes=97,
            gamma=gamma
        )
        finercam.visualize_CAM(finercam_eskimo, image, show=True, alpha=0.5, saving_path=str(visualizations_dir / f'husky_dog_finercam_ref_eskimo_gamma_{gamma}.jpg'))