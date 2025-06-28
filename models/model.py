
# models/model.py
import torch
import torch.nn as nn
from torchvision import models


def load_model(num_classes: int,
               pretrained: bool = True,
               feature_extract: bool = False):
    """
    Load ResNet-50 model, optionally feature-extracting or finetuning
    on a new classification head with `num_classes` outputs.
    """
    model = models.resnet50(pretrained=pretrained)
    # Freeze layers if feature_extract
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(model: nn.Module,
                dataloaders: dict,
                criterion,
                optimizer,
                device: torch.device,
                num_epochs: int = 25):
    """
    Train and validate the model, returning the best model by validation accuracy.
    dataloaders: {'train': train_loader, 'val': val_loader}
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Backward + optimize only if in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()

    print(f"Best val Acc: {best_acc:.4f}")
    # load best weights
    model.load_state_dict(best_model_wts)
    return model

# End of initial modules