import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from matplotlib.widgets import Slider

def plot_metrics(runs,
                 metrics=('loss', 'accuracy', 'recall', 'precision', 'f1_score'),
                 phases=('train', 'val'),
                 saving_dir = None,
                 show = True):
    """
    Plot one figure per metric, overlaying multiple runs and phases.

    Args:
        runs (dict): mapping run name -> CSV filepath or pandas.DataFrame
        metrics (tuple of str): which columns to plot vs. epoch
        phases (tuple of str): which phases to include ('train', 'val', etc.)

    Usage:
        plot_metrics({
            'Run A': 'metrics_runA.csv',
            'Run B': pd.read_csv('metrics_runB.csv')
        }, metrics=('loss','accuracy'), phases=('train','val'))
    """
    for metric in metrics:
        plt.figure()  # one figure per metric
        for name, data in runs.items():
            # load if needed
            if isinstance(data, str):
                df = pd.read_csv(data)
            else:
                df = data.copy()
            for phase in phases:
                sub = df[df['phase'] == phase]
                if sub.empty:
                    continue
                plt.plot(sub['epoch'], sub[metric],
                         label=f"{name}")
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_',' ').title())
        plt.title(f"{metric.replace('_',' ').title()} over Epochs")
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        if saving_dir:
            os.makedirs(saving_dir, exist_ok=True)
            phases_str = "_".join(phases)
            plt.savefig(os.path.join(saving_dir, f"{metric}_{phases_str}.png"))
            print(f"Saved {metric} plot to {saving_dir}")

def cosine_similarity_matrix(weights, device):
    """
    Compute cosine similarity matrix between class weights.
    
    Args:
        weights (torch.Tensor): shape (num_classes, feature_dim)
    
    Returns:
        torch.Tensor: shape (num_classes, num_classes), values in [-1, 1]
    """

    weights = weights.to(device)

    normed = F.normalize(weights, p=2, dim=1)
    return torch.matmul(normed, normed.T)


def plot_top_similar_pairs(sim_matrix: torch.Tensor, top_n=10, class_names=None, show = True, saving_dir=None):
    """
    Plot histogram of the top N most similar class pairs based on cosine similarity,
    with each bar uniquely colored.

    Args:
        sim_matrix (torch.Tensor): shape (num_classes, num_classes), cosine similarity matrix
        top_n (int): number of top similar pairs to plot
        class_names (list[str], optional): class names for label readability
    """
    sim_matrix = sim_matrix.cpu().numpy()
    num_classes = sim_matrix.shape[0]

    # Extract upper triangle (excluding diagonal)
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            pairs.append(((i, j), sim_matrix[i, j]))

    # Sort and get top-N
    top_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

    # Prepare data
    labels = [f"{class_names[i]}-{class_names[j]}" if class_names else f"{i}-{j}" 
              for (i, j), _ in top_pairs]
    values = [score for _, score in top_pairs]

    # Assign distinct colors using colormap
    cmap = plt.get_cmap('tab10') if top_n <= 10 else plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(top_n)]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors, edgecolor='black')
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Class Index Pairs")
    plt.title(f"Top {top_n} Most Similar Class Pairs")
    plt.tight_layout()

    if show:
        plt.show()
    if saving_dir:
        os.makedirs(saving_dir, exist_ok=True)
        plt.savefig(os.path.join(saving_dir, f"top_{top_n}_similar_pairs.png"))
        print(f"Saved top {top_n} similar pairs plot to {saving_dir}")


def compute_relative_confidence_drop(model, input_tensor, cam, target_class, similar_class, top_k=20):
    """
    Compute Relative Confidence Drop as defined in FinerCAM.

    Args:
        model (torch.nn.Module): trained model
        input_tensor (torch.Tensor): shape (1, 3, H, W)
        cam (np.ndarray): 2D saliency map from FinerCAM (H, W), normalized to [0, 1]
        target_class (int): predicted class index
        similar_class (int): most similar class index
        top_k (float): percentage of top pixels to mask (e.g., 20 means top 20%)

    Returns:
        float: relative confidence drop score
    """
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pc = probs[0, target_class].item()
        pd = probs[0, similar_class].item()

    # Create binary mask for top-k% of CAM
    threshold = np.percentile(cam, 100 - top_k)
    cam_mask = cam >= threshold  # shape (H, W)

    # Resize mask to input size
    cam_mask_tensor = torch.tensor(cam_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
    cam_mask_tensor = torch.nn.functional.interpolate(
        cam_mask_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
    )

    # Apply mask (zero out top regions)
    masked_input = input_tensor * (1 - cam_mask_tensor)

    with torch.no_grad():
        masked_logits = model(masked_input)
        masked_probs = torch.softmax(masked_logits, dim=1)
        pc_star = masked_probs[0, target_class].item()
        pd_star = masked_probs[0, similar_class].item()

    # Compute Relative Confidence Drop
    RD = (pc - pc_star) - (pd - pd_star)
    return RD


def compute_deletion_curve(model, input_tensor, cam, target_class, reference_class=None, steps=20):
    """
    Compute and plot a deletion curve for a saliency map.
    
    Args:
        model (torch.nn.Module): trained model
        input_tensor (torch.Tensor): input image, shape (1, 3, H, W)
        cam (np.ndarray): 2D saliency map, normalized to [0, 1]
        target_class (int): index of the predicted target class
        reference_class (int, optional): index of similar reference class to also track
        steps (int): number of deletion steps
    """
    model.eval()
    H, W = cam.shape
    cam_flat = cam.flatten()
    pixel_order = np.argsort(-cam_flat)  # top-importance first
    num_pixels = H * W

    deletion_percents = np.linspace(0, 1, steps + 1)
    target_scores = []
    reference_scores = [] if reference_class is not None else None

    for percent in deletion_percents:
        k = int(percent * num_pixels)
        mask = np.ones(num_pixels, dtype=np.float32)
        mask[pixel_order[:k]] = 0  # mask top-k% pixels
        mask = mask.reshape(H, W)

        # Convert to tensor & resize
        mask_tensor = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(input_tensor.device)
        mask_tensor = F.interpolate(mask_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)

        masked_input = input_tensor * mask_tensor

        with torch.no_grad():
            logits = model(masked_input)
            probs = torch.softmax(logits, dim=1)
            target_scores.append(probs[0, target_class].item())
            if reference_class is not None:
                reference_scores.append(probs[0, reference_class].item())

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(deletion_percents * 100, target_scores, label=f"Target class {target_class}", linewidth=2)
    if reference_scores:
        plt.plot(deletion_percents * 100, reference_scores, label=f"Reference class {reference_class}", linewidth=2, linestyle='--')

    plt.xlabel("Percentage of Top Pixels Removed")
    plt.ylabel("Confidence")
    plt.title("Deletion Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "percent_removed": deletion_percents,
        "target_scores": target_scores,
        "reference_scores": reference_scores
    }



def interactive_mask_slider_script(model, input_tensor, cam, target_class):
    """
    Launch an interactive Matplotlib window with a slider to mask top-k% CAM pixels.
    
    Args:
        model (torch.nn.Module): trained model
        input_tensor (torch.Tensor): input image, shape (1, 3, H, W)
        cam (np.ndarray): 2D saliency map normalized to [0, 1]
        target_class (int): predicted class index
    """
    model.eval()
    cam_flat = cam.flatten()
    pixel_order = np.argsort(-cam_flat)
    num_pixels = cam.size
    H, W = cam.shape

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_ax = ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    ax.axis('off')
    title = ax.set_title("")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Top % Removed', 0, 100, valinit=0, valstep=1)

    def update(val):
        percent = slider.val
        k = int((percent / 100) * num_pixels)
        mask = np.ones(num_pixels, dtype=np.float32)
        mask[pixel_order[:k]] = 0
        mask = mask.reshape(cam.shape)

        # Resize mask and apply
        mask_tensor = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(input_tensor.device)
        mask_tensor = F.interpolate(mask_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        masked_input = input_tensor * mask_tensor

        with torch.no_grad():
            logits = model(masked_input)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, target_class].item()

        # Convert to PIL image
        vis_img = masked_input.clone().squeeze().cpu()
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
        vis_img = to_pil_image(vis_img)

        img_ax.set_data(np.array(vis_img))
        title.set_text(f"{percent:.0f}% Removed - Confidence: {confidence:.4f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)  # Initial draw
    plt.show()

if __name__ == "__main__":
    
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / 'results'
    results_dir = str(results_dir) + os.sep 

    runs = {
        'training': results_dir + 'metrics_train.csv',
        'validation': results_dir + 'metrics_val.csv'
    }
    plot_metrics(runs, metrics=('loss', 'accuracy', 'recall', 'precision', 'f1_score'), phases=('train', 'val', ''), show = False, saving_dir= results_dir + 'plots')