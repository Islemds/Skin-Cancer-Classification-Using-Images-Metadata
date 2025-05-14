######### Save the model ##########
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    ElasticTransform, OpticalDistortion, RandomBrightnessContrast
)
from typing import List
import torchvision
from torch.utils.data import Dataset, DataLoader

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
data_transform = weights.transforms()

images_path = Path("data/")
test_images_path = r'data\test'
test_metadata_path = r'data\test_data.csv'

from data_setup import SkinDataset

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith('.pth') or model_name.endswith('pt'), "Model name should end with .pth or .pt"
    
    model_save_path = target_dir_path / model_name
    
    print(f"Saving model to: {model_save_path}")
    
    torch.save(obj=model.state_dict(), f=model_save_path)
    

######### Load the model #############
def load_model(model_class, 
               target_dir: str, 
               model_name: str, 
               device: torch.device, 
               num_classes: int = 2,
               fusion_operation: str = "mul"):
    model_save_path = Path(target_dir) / model_name
    
    loaded_model = model_class(num_classes=num_classes, fusion_operation=fusion_operation)
    
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    loaded_model = loaded_model.to(device)
    
    print(f"Loaded model from: {model_save_path} with fusion type: {fusion_operation}")
    return loaded_model


############# Plot function #########################

def plot_curves_on_training_validation_data(data: pd.DataFrame, title: str = "Training & Validation Curves"):
    train_loss = data["TrainLoss"]
    train_accuracy = data["TrainAccuracy"]
    train_auc = data["TrainAUC"]
    
    val_loss = data["TestLoss"]
    val_accuracy = data["TestAccuracy"]
    val_auc = data["TestAUC"]
    
    epochs = range(data.shape[0])

    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16, fontweight='bold') 

    # Courbe de Loss
    plt.subplot(1, 3, 1)  
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Courbe d'Accuracy
    plt.subplot(1, 3, 2) 
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    # Courbe d'AUC
    plt.subplot(1, 3, 3) 
    plt.plot(epochs, train_auc, label='Train AUC')
    plt.plot(epochs, val_auc, label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_curves_on_training_data(data: pd.DataFrame, title: str = "Training Curves"):
    loss = data["Loss"]
    accuracy = data["Accuracy"]
    auc = data["AUC"]
    
    epochs = range(data.shape[0])

    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16, fontweight='bold')

    # Courbe de Loss
    plt.subplot(1, 3, 1)  
    plt.plot(epochs, loss, label='Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Courbe d'Accuracy
    plt.subplot(1, 3, 2) 
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    # Courbe d'AUC
    plt.subplot(1, 3, 3) 
    plt.plot(epochs, auc, label='AUC')
    plt.title('AUC')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


########## MAke prediction #############


def predict_and_visualize(test_metadata, index, data_transform, loaded_model, device):
    """
        Predicts the class of a single sample  using an image and its metadata, then visualizes the result.


    Args:
        test_metadata (_type_): _description_
        index (_type_): _description_
        data_transform (_type_): _description_
        loaded_model (_type_): _description_
        device (_type_): _description_
    """
    
    # Extract metadata for the given index
    df_sample = test_metadata.loc[[index]].reset_index(drop=True)
    image_path = df_sample.loc[0, "DDI_path"]
    true_label = df_sample.loc[0, "malignant"]
    skin_tone = list(df_sample.loc[0, ["skin_tone_12", "skin_tone_34", "skin_tone_56", "Disease_Group_Non_melanoma","Disease_Group_melanoma"]])

    single_image = Image.open(image_path).convert("RGB")  
    transformed_image = data_transform(single_image)

    # Convert skin tone features to tensor
    dummy_skin_features = torch.tensor(skin_tone, dtype=torch.float32) 

    # Prepare inputs for model
    transformed_image = transformed_image.unsqueeze(0).to(device)  
    dummy_skin_features = dummy_skin_features.unsqueeze(0).to(device)

    loaded_model.eval()
    with torch.inference_mode():
        outputs = loaded_model(transformed_image, dummy_skin_features)  
        probabilities = torch.softmax(outputs, dim=1)
        positive_class_prob = probabilities[0, 1].item()
        predicted_class = 1 if positive_class_prob > 0.5 else 0

    print(f"True Label: {true_label} | Predicted Class: {predicted_class}") 

    image = np.array(single_image) / 255.0  
    plt.imshow(image)
    plt.title(f"True Label: {true_label} | Predicted Class: {predicted_class} | Skin Tone: {skin_tone}")
    plt.axis("off")
    plt.show()

def predict_test_set(model, test_metadata, device, skin_tone_column=None):

    if skin_tone_column:
        skin_tone_df = test_metadata[test_metadata[skin_tone_column] == 1].reset_index(drop=True)
        skin_tone_dataset = SkinDataset(test_images_path, skin_tone_df, transform=data_transform)
        dataloader = DataLoader(skin_tone_dataset, batch_size=16, shuffle=False)
    else:
        skin_tone_df = test_metadata.reset_index(drop=True)
        test_metadata_dataset = SkinDataset(test_images_path, test_metadata, transform=data_transform)
        dataloader = DataLoader(test_metadata_dataset, batch_size=16, shuffle=False)
    
    model.to(device)
    model.eval()  
        
    all_predictions = []
    all_labels = []
    
    with torch.inference_mode():
        for images, skin_features, labels in dataloader:
            images, skin_features, labels = images.to(device), skin_features.to(device), labels.to(device)
            
            outputs = model(images, skin_features)
            
            probabilities = torch.softmax(outputs, dim=1)
            
            positive_class_probs = probabilities[:, 1]
            
            all_predictions.extend(positive_class_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = (predicted_classes == true_labels).mean()
    auc = roc_auc_score(true_labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc
    }
    
    return predicted_classes, true_labels, metrics



######## TTA #####################


def horizontalFlip(image):
    image_aug = HorizontalFlip(p=1)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

def verticalFlip(image):
    image_aug = VerticalFlip(p=1)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

def transpose(image):
    image_aug = Transpose(p=1)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

def hueSaturationValue(image):
    image_aug = HueSaturationValue(p=1, hue_shift_limit=100, sat_shift_limit=100, val_shift_limit=50)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

def elasticTransform(image):
    image_aug = ElasticTransform(p=1)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

def opticalDistortion(image):
    image_aug = OpticalDistortion(p=1, distort_limit=3, shift_limit=0.4)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

def randomBrightnessContrast(image):
    image_aug = RandomBrightnessContrast(p=1, brightness_limit=0.5, contrast_limit=0.4)
    augmented_image = image_aug(image=image)['image']
    return augmented_image

augmentation_funcs = [
    horizontalFlip, 
    verticalFlip, 
    transpose, 
    hueSaturationValue,
    elasticTransform,
    opticalDistortion,
    randomBrightnessContrast
]




def tta_predict(
    model: torch.nn.Module, 
    image_path: str, 
    augmentation_funcs: List, 
    skin_features: torch.Tensor, 
    device: torch.device
):
    
    original_image = Image.open(image_path).convert("RGB")

    tta_predictions = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = data_transform(original_image).unsqueeze(dim=0).to(device)
        original_pred = model(transformed_image, skin_features.to(device))
        tta_predictions.append(torch.softmax(original_pred, dim=1))

    for aug_func in augmentation_funcs:
        aug_image_np = aug_func(np.array(original_image))
        aug_image = Image.fromarray(aug_image_np)

        with torch.inference_mode():
            transformed_aug_image = data_transform(aug_image).unsqueeze(dim=0).to(device)
            aug_pred = model(transformed_aug_image, skin_features.to(device))
            tta_predictions.append(torch.softmax(aug_pred, dim=1))

    final_pred_probs = torch.mean(torch.stack(tta_predictions), dim=0)

    return final_pred_probs


import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_skin_tone(loaded_model, test_metadata,device, skin_tone_column=None):
    if skin_tone_column:
        skin_tone_df = test_metadata[test_metadata[skin_tone_column] == 1].reset_index(drop=True)
    else:
        skin_tone_df = test_metadata.reset_index(drop=True)
        
    acc = 0
    all_true_labels = []
    all_pred_probs = []
    predicted_labels = []
    
    for index, row in skin_tone_df.iterrows():
        image_path = row["DDI_path"]
        true_label = row["malignant"]
        dummy_skin_features = torch.tensor(
            [row["skin_tone_12"], row["skin_tone_34"], row["skin_tone_56"], row["Disease_Group_Non_melanoma"], row["Disease_Group_melanoma"]], 
            dtype=torch.float32
        ).unsqueeze(0)
        
        final_pred_probs = tta_predict(loaded_model, image_path, augmentation_funcs, dummy_skin_features, device)
        predicted_class = torch.argmax(final_pred_probs, dim=1).item()
        
        if predicted_class == true_label:
            acc += 1
        
        all_true_labels.append(true_label)
        all_pred_probs.append(final_pred_probs[0, 1].item())
        predicted_labels.append(predicted_class)
    
    accuracy = acc / len(skin_tone_df) if len(skin_tone_df) > 0 else 0
    auc_score = roc_auc_score(all_true_labels, np.array(all_pred_probs)) if all_true_labels else 0
    
    metrics = {"accuracy": accuracy, "auc": auc_score}
    
    
    return all_true_labels, predicted_labels, metrics
