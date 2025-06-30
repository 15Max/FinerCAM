import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


    
class MethodsCAM:
    """
    Abstract base class for CAM methods.
    Registers hooks to capture activations and gradients from a target layer during forward and backward passes.
    It also includes methods for preprocessing images and visualizing the CAM overlay on the original image.
    """
    def __init__(self, model, target_layer):
        """
        Initialize the CAM method with the model and target layer.
        Args:
            model (torch.nn.Module): The model to be used for generating CAM.
            target_layer (torch.nn.Module): The layer from which to extract activations and gradients.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        """
        Hook to capture the gradients during the backward pass.
        Args:
            module (torch.nn.Module): The layer to which the hook is attached.
            grad_input (tuple): The gradients with respect to the inputs of the layer.
            grad_output (tuple): The gradients with respect to the outputs of the layer.
        """
        self.gradients = grad_output[0].detach()

    def forward_hook(self, module, input, output):
        """
        Hook to capture the activations during the forward pass.
        Args:
            module (torch.nn.Module): The layer to which the hook is attached.
            input (tuple): The inputs to the layer.
            output (torch.Tensor): The outputs of the layer.
        """
        self.activations = output.detach()

    @staticmethod
    def preprocess_image(image_path, device):
        """
        Preprocess the input image for the model according to the standard ImageNet preprocessing.
        Args:
            image_path (str): Path to the input image.
            device (torch.device): The device to which the image tensor will be moved.
        Returns:
            image_tensor (torch.Tensor): The preprocessed image tensor ready for model input.
            original_image (PIL.Image): The original image for visualization purposes.
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        return image_tensor, image
    

    @staticmethod
    def visualize_CAM(cam, original_image, alpha=0.5, show = False, saving_path = None):
        """
        Visualize the Class Activation Map (CAM) overlay on the original image.
        Args:
            cam (np.ndarray): The Class Activation Map to be visualized.
            original_image (PIL.Image): The original image on which the CAM will be overlaid.
            alpha (float): The transparency factor for the overlay (0 = transparent, 1 = opaque).
            show (bool): Whether to display the overlay image using matplotlib.
        Returns:
            overlay_image (PIL.Image): The image with the CAM overlay applied.
        """
        cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
        cam_colored = np.array(cam_resized)
        cam_colored = plt.cm.jet(cam_colored.astype(np.uint8))[:, :, :3]  # Apply color map
        cam_colored = (cam_colored * 255).astype(np.uint8)

        overlay = np.array(original_image) * (1 - alpha) + cam_colored * alpha
        overlay = overlay.astype(np.uint8)

        if show:
            plt.imshow(overlay)
            plt.axis('off')
            plt.title("CAM Overlay with Alpha = {}".format(alpha))
            plt.show()
        
        if saving_path is not None:
            overlay_image = Image.fromarray(overlay)
            overlay_image.save(saving_path)

        return Image.fromarray(overlay)
    


class GradCAM(MethodsCAM):
    def __init__(self, model, target_layer):
        """
        Initialize the GradCAM method with the model and target layer.
        Args:
            model (torch.nn.Module): The model to be used for generating GradCAM.
            target_layer (torch.nn.Module): The layer from which to extract activations and gradients.
        """
        super().__init__(model, target_layer)
        self.gradients = None
        self.activations = None


    def generate_CAM(self, input_image, target_class=None):
        """
        Generate the Class Activation Map (CAM) using GradCAM.
        Args:
            input_image (torch.Tensor): The preprocessed input image tensor.
            target_class (int, optional): The class index for which to generate the CAM. If None, the predicted class will be used.
        Returns:
            cam (np.ndarray): The generated Class Activation Map as a numpy array.
        """
        
        model_output = self.model(input_image)  # Forward pass
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()  # Use the predicted class if none specified

        # Backpropagate to get gradients w.r.t. the target class
        self.model.zero_grad()  # Zero the gradients
        class_loss = model_output[:, target_class]  # Get the output score for the target class
        class_loss.backward(retain_graph=True)  # Backward pass

        # Generate CAM by weighting the activations by the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # Global average pooling over the gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]  # Weight each activation map

        cam = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()  # Average the weighted activations to get the heatmap
        cam = np.maximum(cam, 0)  # Apply ReLU to the heatmap
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize the heatmap to [0, 1]

        return cam
    
class FinerCAM(MethodsCAM):
    def __init__(self, model, target_layer):
        """
        Initialize the FinerCAM method with the model and target layer.
        Args:
            model (torch.nn.Module): The model to be used for generating FinerCAM.
            target_layer (torch.nn.Module): The layer from which to extract activations and gradients.
        """
        super().__init__(model, target_layer)
        self.gradients = None
        self.activations = None
    

    def generate_CAM(self, input_image, target_class=None, reference_classes=None, gamma=0.6):
        """
        Generate the Class Activation Map (CAM) using FinerCAM with one or more reference classes.
        
        Args:
            input_image (torch.Tensor): The preprocessed input image tensor.
            target_class (int, optional): The target class index. If None, use model prediction.
            reference_classes (list[int] or int, optional): One or more reference class indices. 
                If None, use the next class as default.
            gamma (float): The strength of comparison against reference classes.
        
        Returns:
            np.ndarray: The generated Class Activation Map as a 2D NumPy array in [0, 1].
        """
        model_output = self.model(input_image)

        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        if reference_classes is None:
            reference_classes = [(target_class + 1) % model_output.shape[1]]
        elif isinstance(reference_classes, int):
            reference_classes = [reference_classes]

        # Get logits
        y_c = model_output[:, target_class]
        y_d_total = sum([model_output[:, d] for d in reference_classes])
        y_diff = y_c - gamma * y_d_total / len(reference_classes)

        # Zero gradients
        self.model.zero_grad()
        y_diff.backward(retain_graph=True)

        # Compute pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = self.activations.clone()
        for i in range(pooled_gradients.shape[0]):
            weighted_activations[:, i, :, :] *= pooled_gradients[i]

        cam = torch.mean(weighted_activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
