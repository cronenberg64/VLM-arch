import torch
from torchvision import transforms
from timm.data import create_transform

def build_transforms(cfg, is_train=True):
    """
    Builds the data transformation pipeline.
    Uses timm's create_transform for advanced augmentations if specified.
    """
    input_size = cfg.dataset.image_size
    
    if is_train:
        # Check if we want to use timm's advanced augmentations
        # For small data, standard augmentations might be better than heavy ones like RandAugment
        # unless we have strong regularization needs.
        # We'll default to standard CIFAR-style augs for now, but allow expansion.
        
        t_list = []
        if input_size != 32:
            t_list.append(transforms.Resize(input_size))
            
        t_list.extend([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transforms.Compose(t_list)
    else:
        t_list = []
        if input_size != 32:
            t_list.append(transforms.Resize(input_size))
            
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transforms.Compose(t_list)
