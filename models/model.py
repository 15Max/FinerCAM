# models/model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from torch import amp
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
