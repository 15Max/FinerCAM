# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from collections import defaultdict
import numpy as np
from torch import amp
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import os  


def load_model(num_classes: int,
               weights =ResNet50_Weights.DEFAULT,
               feature_extract: bool = False):
    """
    Load ResNet-50 model, optionally feature-extracting or finetuning
    on a new classification head with `num_classes` outputs.

    """
    model = resnet50(weights=weights)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(model: nn.Module,
                dataloaders: dict,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_epochs: int = 25,
                results_dir: str = None,
                model_save_path: str = None,
                scheduler=None,
                grad_clip: float = None,
                early_stop: bool = False,
                patience: int = 5,
                min_delta: float = 0.005):
    """
    Train and validate model
    Args:
        model (nn.Module): model to train.
        dataloaders (dict): {'train', 'val'} DataLoaders.
        criterion (nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer.
        device (torch.device): CPU or GPU.
        num_epochs (int): number of epochs.
        results_dir (str): directory to save metrics CSVs.
        model_save_path (str): filepath to save best model.
        scheduler: LR scheduler (optional).
        grad_clip (float): max norm for grads (optional).
        early_stop (bool): enable early stopping.
        patience (int): epochs to wait for improvement.
        min_delta (float): minimal val-acc improvement to reset patience.
    Returns:
        model (nn.Module): model loaded with best weights.
    """

    # Mixed precision setup
    scaler = amp.GradScaler() if device.type == 'cuda' else None

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0

    records = []

    model.to(device)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        for phase in ('train', 'val'):
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            all_preds, all_labels = [], []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward (with optional mixed precision)
                with amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if is_train:
                    # backward + optimize
                    if scaler:
                        scaler.scale(loss).backward()
                        if grad_clip:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            # epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            current_lr = optimizer.param_groups[0]['lr']

            print(f"{phase} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | "
                  f"prec: {precision:.4f} | rec: {recall:.4f} | f1: {f1:.4f} | lr: {current_lr:.6f}")

            records.append({
                'epoch': epoch,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'lr': current_lr
            })

            # validation‐only logic
            if phase == 'val':
                if scheduler is not None:
                    scheduler.step(epoch_loss)

                improved = (epoch_acc - best_acc) > min_delta
                if improved:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_weights = model.state_dict().copy()
                    no_improve = 0
                    if model_save_path:
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                        torch.save(best_weights, model_save_path)
                        print(f"Saved best model to {model_save_path}")
                else:
                    no_improve += 1

                if early_stop and (no_improve >= patience):
                    print(f"Early stopping at epoch {epoch}. No improvement in {patience} epochs.")
                    model.load_state_dict(best_weights)
                    # save metrics and return
                    if results_dir:
                        os.makedirs(results_dir, exist_ok=True)
                        df = pd.DataFrame(records)
                        # train vs val CSVs
                        train_csv = os.path.join(results_dir, 'metrics_train.csv')
                        val_csv   = os.path.join(results_dir, 'metrics_val.csv')
                        df[df.phase=='train'].to_csv(train_csv, index=False)
                        df[df.phase=='val'].to_csv(val_csv,   index=False)
                        print(f"Metrics saved to {results_dir}")
                    return model

    # end of epochs: restore best
    model.load_state_dict(best_weights)
    print(f"Training complete. Best val Acc: {best_acc:.4f} at epoch {best_epoch}")

    # save full metrics if requested
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(records)
        train_csv = os.path.join(results_dir, 'metrics_train.csv')
        val_csv   = os.path.join(results_dir, 'metrics_val.csv')
        df[df.phase=='train'].to_csv(train_csv, index=False)
        df[df.phase=='val'].to_csv(val_csv,   index=False)
        print(f"Metrics saved to {results_dir}")

    return model


def evaluate_model(weights_dir: str,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  class_names: list,
                  results_dir: str):
    """
    Evaluate a model on the test set and save summary metrics + confusion matrix to CSV.

    Args:
        model (torch.nn.Module): Trained model (in eval mode or not).
        test_loader (DataLoader): DataLoader for the test split.
        device (torch.device): Device for computation.
        class_names (list): List of class labels.
        results_dir (str): Path to a directory to save results.

    Writes:
        1) A summary CSV with one row: accuracy, precision, recall, f1.
        2) A confusion-matrix CSV alongside, named `<base>_confusion_matrix.csv`.
    """
    # Ensure output dir exists
    os.makedirs(os.path.dirname(results_dir), exist_ok=True)

    model = load_model(
        num_classes=len(class_names),
        weights=None,  
        feature_extract=False
    )

    checkpoint = torch.load(weights_dir, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    # Move model to device & eval
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute summary metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Write summary
    summary_df = pd.DataFrame([{
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }])
    summary_df.to_csv(results_dir + '/metrics_test.csv', index=False)

    # Compute & write confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    base, _ = os.path.splitext(results_dir)
    cm_csv = f"{base}/confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def class_uncertainty(model: torch.nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   class_names: list[str],
                                   device: torch.device):
    """
    For each true class, compute:
      - avg softmax confidence in the correct label
      - avg entropy of the full softmax distribution
      - number of samples
      - number of correct top-1 predictions
      - per-class accuracy

    Returns a DataFrame sorted by ascending avg_confidence (most uncertain first).
    """
    model.to(device).eval()

    # accumulators keyed by true class idx
    confs = defaultdict(list)
    ents  = defaultdict(list)
    correct_counts = defaultdict(int)
    total_counts   = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Gathering stats"):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)              # [B, C]
            probs  = F.softmax(logits, dim=1)   # [B, C]
            logp   = F.log_softmax(logits, dim=1)

            # per-sample scalar metrics
            pred_idxs   = logits.argmax(dim=1)
            conf_true   = probs.gather(1, labels.unsqueeze(1))[:,0]
            entropies   = -(probs * logp).sum(dim=1)

            for lbl, pred, c, e in zip(labels.cpu(), 
                                       pred_idxs.cpu(), 
                                       conf_true.cpu(), 
                                       entropies.cpu()):
                lbl_i = int(lbl)
                total_counts[lbl_i] += 1
                if int(pred) == lbl_i:
                    correct_counts[lbl_i] += 1

                confs[lbl_i].append(float(c))
                ents[lbl_i].append(float(e))

    # build DataFrame
    records = []
    for idx, name in enumerate(class_names):
        n_total = total_counts.get(idx, 0)
        if n_total == 0:
            continue
        n_correct = correct_counts.get(idx, 0)
        records.append({
            'class_idx':      idx,
            'class_name':     name,
            'n_samples':      n_total,
            'n_correct':      n_correct,
            'accuracy':       n_correct / n_total,
            'avg_confidence': np.mean(confs[idx]),
            'avg_entropy':    np.mean(ents[idx]),
        })

    df = pd.DataFrame.from_records(records)
    # sort by uncertainty (low confidence first)
    df = df.sort_values('avg_confidence').reset_index(drop=True)
    return df