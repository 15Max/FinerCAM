import os  
import torch  
from torch.utils.data import DataLoader, Subset 
from torchvision import transforms, datasets  
from sklearn.model_selection import train_test_split  
import numpy as np


def get_dataloaders(data_dir: str,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.1,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    input_size: int = 224,  
                    seed: int = None):
    """
    Create stratified dataloaders for training, validation, and testing
    from an image-folder dataset.

    Args:
        data_dir (str): Root directory with subfolders per class (e.g n02085620-Chihuahua)
        train_ratio (float): Fraction of data for training (default 0.8).
        val_ratio (float): Fraction for validation (default 0.1).
        batch_size (int): Number of samples per batch.
        num_workers (int): Parallel workers for data loading.
        input_size (int): Resize shorter side of image to this size. (default 224 for ResNet50)
        seed: Random seed for reproducible splits (Optional).

    Returns:
        loaders (dict): {'train', 'val', 'test'} DataLoader objects.
        class_names (List[str]): Names of classes inferred from folder names.
    """


    # Get the seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Define image transforms
    # common_transforms: applied to all sets (resize, tensor, normalize)
    # resize to input_size, convert to tensor, normalize with ImageNet stats
    # Needed for the ResNet50 model
    common_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),   # fixed spatial size
        transforms.ToTensor(),                         # HxWxC [0,255] -> CxHxW [0.0,1.0]
        transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet mean (IMAGENET1K_V2)
                             [0.229, 0.224, 0.225])    # ImageNet std  (IMAGENET1K_V2)
    ])
    
    # Some augmentation for training data to act as a regularizer
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),             # data augmentation
        transforms.ToTensor(),                         # convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset and get labels
    # ImageFolder maps each subfolder to a class index
    # samples list contains (filepath, class_idx)

    full_dataset = datasets.ImageFolder(root=data_dir, transform=common_transforms)
    class_names = full_dataset.classes
    # Drop the prefix before the '-' to clean up names, e.g. 'Chihuahua'
    class_names = [name.split('-', 1)[-1] for name in class_names] 


    # Prepare indices and labels for splitting
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    # Split: first into train vs temp (val+test), then temp into val vs test
    # Stratify to preserve class proportions in each subset


    # Train vs (Val+Test)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed
    )

    # Validation vs Test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp,
        train_size=val_ratio,
        stratify=y_temp,
        random_state=seed 
    )

    # Wrap subsets and assign transforms
    # Subset reuses full_dataset; we just point to different indices
    # Only training subset gets the augmentation transform


    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    train_dataset.dataset.transform = train_transform

    # Create DataLoader objects for each subset
    # shuffle train, keep val/test in order

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    # Return the three loaders and class mapping
    return {'train': train_loader,
            'val': val_loader,
            'test': test_loader}, class_names


###################################################
################# TESTING #########################
###################################################


if __name__ == "__main__":

    from pathlib import Path

    # Go up one dir

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / 'data' / 'images'
   
    print(f"Using data directory: {data_dir}")

    # Get the dataloaders
    dataloaders, class_names = get_dataloaders(
        data_dir=str(data_dir),
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=32,
        num_workers=4,
        input_size=224)
    
    print(f"Class names: {class_names}")
    print(f"Train loader: {len(dataloaders['train'])} batches")
    print(f"Validation loader: {len(dataloaders['val'])} batches")
    print(f"Test loader: {len(dataloaders['test'])} batches")