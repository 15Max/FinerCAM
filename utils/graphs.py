import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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

def plot_top_similar_pairs(sim_matrix: torch.Tensor, top_n=10, save_path=None, show=True):
    """
    Plot histogram of the top N most similar class pairs based on cosine similarity.

    Args:
        sim_matrix (torch.Tensor): shape (num_classes, num_classes), cosine similarity matrix
        top_n (int): number of top similar pairs to plot
    """
    sim_matrix = sim_matrix.cpu().numpy()
    num_classes = sim_matrix.shape[0]

    # Collect upper triangle (excluding diagonal)
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            pairs.append(((i, j), sim_matrix[i, j]))

    # Sort by similarity descending
    top_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

    # Prepare labels and values
    labels = [f"{i}-{j}" for (i, j), _ in top_pairs]
    values = [score for _, score in top_pairs]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="skyblue")
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Class Index Pairs")
    plt.title(f"Top {top_n} Most Similar Class Pairs")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if show:
        plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)




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