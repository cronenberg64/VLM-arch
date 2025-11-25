import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig

class DataModule:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root = cfg.dataset.root
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.pin_memory = cfg.training.pin_memory
        self.dataset_name = cfg.dataset.name.lower()
        self.subset_size = cfg.dataset.subset
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_transforms(self, train: bool = True):
        # Basic transforms - will be enhanced in transforms.py
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def setup(self):
        if self.dataset_name == 'cifar10':
            dataset_cls = datasets.CIFAR10
        elif self.dataset_name == 'cifar100':
            dataset_cls = datasets.CIFAR100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        full_train_dataset = dataset_cls(
            root=self.root, train=True, download=True, transform=self.get_transforms(train=True)
        )
        self.test_dataset = dataset_cls(
            root=self.root, train=False, download=True, transform=self.get_transforms(train=False)
        )

        # Handle subsampling
        if self.subset_size:
            indices = self._get_stratified_subset_indices(full_train_dataset, self.subset_size)
            self.train_dataset = Subset(full_train_dataset, indices)
            self.val_dataset = self.test_dataset 
        else:
            self.train_dataset = full_train_dataset
            self.val_dataset = self.test_dataset

    def _get_stratified_subset_indices(self, dataset, subset_size):
        targets = np.array(dataset.targets)
        classes = np.unique(targets)
        num_classes = len(classes)
        samples_per_class = subset_size // num_classes
        
        indices = []
        for c in classes:
            class_indices = np.where(targets == c)[0]
            # Ensure we don't ask for more samples than exist
            n_sample = min(len(class_indices), samples_per_class)
            indices.extend(np.random.choice(class_indices, n_sample, replace=False))
            
        return indices

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
