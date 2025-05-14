import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os



class SkinDataset(Dataset):
    def __init__(self, 
                 images_path: str, 
                 metadata: pd.DataFrame, 
                 transform : Optional[torch.nn.Module] = None):
        self.images_path = images_path
        self.metadata = metadata.copy()
        self.transform = transform
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            row = self.metadata.iloc[idx]
        except KeyError:
            print(f"KeyError with idx: {idx}")
            raise
        
        # Class folder based on the value of 'malignant'
        if row["malignant"] == 0:
            class_folder = "begin"
        else:
            class_folder = "malignant" 
        img_path = row['DDI_path']
        img_path = img_path.replace("\\", "/")
        
        
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        
        skin_features = torch.tensor([
            row['skin_tone_12'], 
            row['skin_tone_34'], 
            row['skin_tone_56'],
            row['Disease_Group_Non_melanoma'],
            row['Disease_Group_melanoma']
        ], dtype=torch.float)
        

        # Target label
        label = torch.tensor(row['malignant'], dtype=torch.long)
        
        return image, skin_features, label
    



def create_datasets(
    images_path: str,
    metadata: pd.DataFrame,
    transform,
):
    dataset = SkinDataset(images_path = images_path,metadata=metadata, transform = transform)
    
    return dataset