
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

    husky = False
    rhodesian = False

    if (husky):
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
        
        
        husky_dog_with_husky_fm = gradcam.visualize_CAM(cam_husky, image, show = False, alpha = 0.5, saving_path=str(visualizations_dir / 'husky_dog_with_husky_fm.jpg'))

        cam_eskimo = gradcam.generate_CAM(
            input_image=image_tensor,
            target_class=97)
        
        husky_dog_with_eskimo_fm = gradcam.visualize_CAM(cam_eskimo, image, show = False, alpha = 0.5, saving_path=str(visualizations_dir / 'husky_dog_with_eskimo_fm.jpg'))


        finercam = FinerCAM(
            model=model,
            target_layer=target_layer)
        
        finercam_eskimo = finercam.generate_CAM(
            input_image=image_tensor,
            target_class=99,
            reference_classes=97,
            gamma=1
        )

        husky_dog_finercam_ref_eskimo = finercam.visualize_CAM(finercam_eskimo, image, show = False, alpha = 0.5, saving_path=str(visualizations_dir / 'husky_dog_finercam_ref_eskimo.jpg'))

        results = compute_deletion_curve(
            model=model,
            input_tensor=image_tensor,
            cam=finercam_eskimo,
            target_class=99,
            reference_class=97,
            steps=100,
            show=False,
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
            finercam.visualize_CAM(finercam_eskimo, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / f'husky_dog_finercam_ref_eskimo_gamma_{gamma}.jpg'))


        relative_confidence_drop = compute_relative_confidence_drop(
            model=model,
            input_tensor=image_tensor,
            cam=finercam_eskimo,
            target_class=99,
            similar_class=97,
            top_k=50)

        print("Relative confidence drop for husky:", relative_confidence_drop)

    if (rhodesian):


        rhodesian_ridgeback_target = str("data/images/n02087394-Rhodesian_ridgeback/n02087394_2319.jpg")
        redbone_reference = str("data/images/n02090379-redbone/n02090379_5493.jpg")

        rhodesian_ridgeback_index = 8
        redbone_index = 17

        gradcam = GradCAM(
            model=model,
            target_layer=target_layer)

        outputs = model(gradcam.preprocess_image(image_path=rhodesian_ridgeback_target, device=device)[0])
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted class index: {predicted.item()}")

        topk_values, topk_indices = torch.topk(outputs, k=3)
        print("Top 3 class indices:", topk_indices[0].tolist())

        image_tensor, image = gradcam.preprocess_image(image_path=rhodesian_ridgeback_target, device=device)

        cam_ridgeback = gradcam.generate_CAM(
            input_image=image_tensor,
            target_class=rhodesian_ridgeback_index)

        gradcam.visualize_CAM(cam_ridgeback, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / 'ridgeback_with_ridgeback_fm.jpg'))

        cam_redbone = gradcam.generate_CAM(
            input_image=image_tensor,
            target_class=redbone_index)

        gradcam.visualize_CAM(cam_redbone, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / 'ridgeback_with_redbone_fm.jpg'))

        finercam = FinerCAM(
            model=model,
            target_layer=target_layer)

        finercam_redbone = finercam.generate_CAM(
            input_image=image_tensor,
            target_class=rhodesian_ridgeback_index,
            reference_classes=redbone_index,
            gamma=1
        )

        finercam.visualize_CAM(finercam_redbone, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / 'ridgeback_finercam_ref_redbone.jpg'))

        results = compute_deletion_curve(
            model=model,
            input_tensor=image_tensor,
            cam=finercam_redbone,
            target_class=rhodesian_ridgeback_index,
            reference_class=redbone_index,
            steps=100,
            show=False,
        )

        interactive_mask_slider_script(model, image_tensor, finercam_redbone, target_class=rhodesian_ridgeback_index)

        gammas = [0.2,0.4, 0.6, 0.8, 1.0]
        for gamma in gammas:
            finercam_redbone = finercam.generate_CAM(
                input_image=image_tensor,
                target_class=rhodesian_ridgeback_index,
                reference_classes=redbone_index,
                gamma=gamma
            )
            finercam.visualize_CAM(finercam_redbone, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / f'ridgeback_finercam_ref_redbone_gamma_{gamma}.jpg'))

        relative_confidence_drop = compute_relative_confidence_drop(
            model=model,
            input_tensor=image_tensor,
            cam=finercam_redbone,
            target_class=rhodesian_ridgeback_index,
            similar_class=redbone_index,
            top_k=20)

        print("Relative confidence drop for rhodesian ridgeback:", relative_confidence_drop)
    

    irish_wolfhound_idx = 19
    scottish_deerhound_idx = 26

    irish_wolfhound_target = str("data/images/n02090721-Irish_wolfhound/n02090721_1742.jpg")
    scottish_deerhound_reference = str("data/images/n02092002-Scottish_deerhound/n02092002_296.jpg")

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer)

    outputs = model(gradcam.preprocess_image(image_path=irish_wolfhound_target, device=device)[0])
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted class index: {predicted.item()}")

    topk_values, topk_indices = torch.topk(outputs, k=3)
    print("Top 3 class indices:", topk_indices[0].tolist())

    image_tensor, image = gradcam.preprocess_image(image_path=irish_wolfhound_target, device=device)

    cam_wolfhound = gradcam.generate_CAM(
        input_image=image_tensor,
        target_class=irish_wolfhound_idx)

    gradcam.visualize_CAM(cam_wolfhound, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / 'wolfhound_with_wolfhound_fm.jpg'))

    cam_deerhound = gradcam.generate_CAM(
        input_image=image_tensor,
        target_class=scottish_deerhound_idx)

    gradcam.visualize_CAM(cam_deerhound, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / 'wolfhound_with_deerhound_fm.jpg'))

    finercam = FinerCAM(
        model=model,
        target_layer=target_layer)

    finercam_deerhound = finercam.generate_CAM(
        input_image=image_tensor,
        target_class=irish_wolfhound_idx,
        reference_classes=scottish_deerhound_idx,
        gamma=1
    )

    finercam.visualize_CAM(finercam_deerhound, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / 'wolfhound_finercam_ref_deerhound.jpg'))

    results = compute_deletion_curve(
        model=model,
        input_tensor=image_tensor,
        cam=finercam_deerhound,
        target_class=irish_wolfhound_idx,
        reference_class=scottish_deerhound_idx,
        steps=100,
        show=False,
    )

    interactive_mask_slider_script(model, image_tensor, finercam_deerhound, target_class=irish_wolfhound_idx)

    gammas = [0.2, 0.4, 0.6, 0.8, 1.0]
    for gamma in gammas:
        finercam_deerhound = finercam.generate_CAM(
            input_image=image_tensor,
            target_class=irish_wolfhound_idx,
            reference_classes=scottish_deerhound_idx,
            gamma=gamma
        )
        finercam.visualize_CAM(finercam_deerhound, image, show=False, alpha=0.5, saving_path=str(visualizations_dir / f'wolfhound_finercam_ref_deerhound_gamma_{gamma}.jpg'))

    relative_confidence_drop = compute_relative_confidence_drop(
        model=model,
        input_tensor=image_tensor,
        cam=finercam_deerhound,
        target_class=irish_wolfhound_idx,
        similar_class=scottish_deerhound_idx,
        top_k=20)

    print("Relative confidence drop for irish wolfhound:", relative_confidence_drop)