
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.data  import get_dataloaders
from models.model import load_model, train_model, evaluate_model
from pathlib import Path
import argparse


if __name__ == "__main__":
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description='Train a ResNet50 model on image data.')
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--test', action='store_true', help='Enable testing mode')

    args = parser.parse_args()
    training = args.train
    testing = args.test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training mode: {training}, Testing mode: {testing}")

    project_dir = Path(__file__).resolve().parent
    model_save_path = project_dir / 'models' / 'checkpoints' / 'best_resnet50.pth'


    loaders, class_names = get_dataloaders(
        data_dir = 'data/images',
        train_ratio = 0.7,
        val_ratio= 0.1,
        batch_size = 32,
        num_workers = 4,
        input_size = 224,
        seed = 15)

    train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']

    if training:
        print(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches")
        # Model: Resnet50 model, pretrained on ImageNet
        model = load_model(len(class_names), feature_extract=False)

        # Loss: CrossEntropyLoss for multi-class classification
        criterion = nn.CrossEntropyLoss()

        # Optimizer: SGD with momentum
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, momentum=0.9
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',         # we want the val loss to go *down*
            factor=0.1,         # new_lr = old_lr * factor
            patience=3,         # wait this many epochs before reducing
            min_lr=1e-6         # lower bound on the LR
        )


        # Train + save best model
        Res50FT = train_model(
            model,
            {'train': train_loader, 'val': val_loader},
            criterion,
            optimizer,
            device=torch.device('cuda'),
            num_epochs=25,
            results_dir = str(project_dir / 'results'),
            model_save_path = str(model_save_path),
            scheduler = scheduler,
            grad_clip= 1,
            early_stop=True,
            patience=3,
            min_delta=0.005
        )

    if testing:

        if not model_save_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_save_path}. Please train the model first.")

        evaluate_model(
            weights_dir = str(model_save_path),
            test_loader = test_loader,
            device = device,
            class_names = class_names,
            results_dir = str(project_dir / 'results'),
        )