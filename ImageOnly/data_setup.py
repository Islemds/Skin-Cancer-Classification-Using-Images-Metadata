from torchvision import transforms
import torch
from torchvision import transforms, datasets

from pathlib import Path
import os


def create_datasets(
    train_dir: str,
    test_dir: str,
    transform,
):
    train_data = datasets.ImageFolder(root=train_dir, transform = transform)
    test_data = datasets.ImageFolder(root=test_dir, transform = transform)
    num_classes = len(train_data.classes)
    
    return train_data, test_data, num_classes






