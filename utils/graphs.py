import os
import pandas as pd
import matplotlib.pyplot as plt

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