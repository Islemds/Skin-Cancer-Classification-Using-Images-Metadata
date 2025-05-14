
import torch
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import pandas as pd


from albumentations import (
PadIfNeeded,
HorizontalFlip,
VerticalFlip,
Transpose,
HueSaturationValue,
ElasticTransform,
GridDistortion,
OpticalDistortion,
RandomBrightnessContrast,
RandomGamma,Resize
)


device = "cuda" if torch.cuda.is_available() else "cpu"


####### Save the model ######
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith('.pth') or model_name.endswith('pt'), "Model name should end with .pth or .pt"
    
    model_save_path = target_dir_path / model_name
    
    print(f"[INFO] Saving model to: {model_save_path}")
    
    torch.save(obj=model.state_dict(), f=model_save_path)
    
    
    
    
###### Load the model ######
def load_model(model_class, 
               target_dir: str, 
               model_name: str, 
               backbone: str,
               device: torch.device, 
               num_classes: int = 2):
    model_save_path = Path(target_dir) / model_name
    
    loaded_model = model_class(num_classes=num_classes, backbone = backbone)
    
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    loaded_model = loaded_model.to(device)
    
    print(f"[INFO] Loaded model from: {model_save_path}")
    return loaded_model


from torchvision import transforms
from torchvision.models import EfficientNet_B5_Weights, Inception_V3_Weights, DenseNet121_Weights

def get_data_transform(backbone="efficientnet_b5"):
    if backbone.lower() == "efficientnet_b5":
        weights = EfficientNet_B5_Weights.DEFAULT
        return weights.transforms()

    elif backbone.lower() == "inception_v3":
        weights = Inception_V3_Weights.DEFAULT
        return weights.transforms()

    elif backbone.lower() == "densenet121":
        weights = DenseNet121_Weights.DEFAULT
        return weights.transforms()

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")



###### Plot the curves for training #########
def plot_curves_on_training_validation_data(data: pd.DataFrame, title: str = "Training & Validation Curves"):
    train_loss = data["TrainLoss"]
    train_Accuracy = data["TrainAccuracy"]
    train_AUC = data["TrainAUC"]
    train_AUC_pr = data["TrainAUC-PR"]
    
    val_loss = data["TestLoss"]
    val_Accuracy = data["TestAccuracy"]
    val_AUC = data["TestAUC"]
    val_AUC_pr = data["TestAUC-PR"]
    
    epochs = range(data.shape[0])

    plt.figure(figsize=(15, 7))
    plt.suptitle(title, fontsize=16, fontweight='bold') 

    plt.subplot(2, 2, 1)  
    plt.plot(epochs, train_loss, label='TrainLoss')
    plt.plot(epochs, val_loss, label='ValLoss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 2, 2) 
    plt.plot(epochs, train_Accuracy, label='TrainAccuracy')
    plt.plot(epochs, val_Accuracy, label='ValAccuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 2, 3) 
    plt.plot(epochs, train_AUC, label='TrainAUC')
    plt.plot(epochs, val_AUC, label='ValAUC')
    plt.title('AUC')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 2, 4)  
    plt.plot(epochs, train_AUC_pr, label='TrainAUC-PR')
    plt.plot(epochs, val_AUC_pr, label='ValAUC-PR')
    plt.title('AUC_PR')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    
    
    
def plot_curves_on_training_data(data: pd.DataFrame, title: str = "Training Curves"):
    
    loss = data["Loss"]
    Accuracy = data["Accuracy"]
    AUC = data["AUC"]
    AUC_pr = data["AUC-PR"]
    epochs = range(data.shape[0])

    
    plt.figure(figsize=(15, 7))
    plt.suptitle(title, fontsize=16, fontweight='bold')

    plt.subplot(2, 2, 1)  
    plt.plot(epochs, loss, label='Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 2, 2) 
    plt.plot(epochs, Accuracy, label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 2, 3) 
    plt.plot(epochs, AUC, label='AUC')
    plt.title('AUC')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 2, 4)  
    plt.plot(epochs, AUC_pr, label='AUC-PR')
    plt.title('AUC_PR')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()  
    plt.show()



######### Make predictions ################

from PIL import Image
import torch
import torchvision
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

def predict_test_set(model: torch.nn.Module,
              dataset,
              transform: torchvision.transforms = None,
              device: torch.device = torch.device("cpu")):
    
    images_path = list(dataset["DDI_path"])
    true_labels = list(dataset["malignant"])  # Extract true labels from dataset
    predicted_labels = []
    predicted_probs = []
    
    if transform is None:
        weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
        transform = weights.transforms()
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for idx, image_path in enumerate(images_path):
            img = Image.open(image_path).convert("RGB")
            transformed_image = transform(img).unsqueeze(dim=0).to(device)
            
            output = model(transformed_image)
            probabilities = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probabilities, dim=1).item()
            
            predicted_labels.append(pred_label)
            predicted_probs.append(probabilities[0, 1].item())  # Probabilité de la classe "malignant"
    
    # Calcul des métriques
    accuracy = accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs) if len(set(true_labels)) > 1 else None  # AUC nécessite au moins 2 classes distinctes
    
    metrics = {
        "accuracy": accuracy,
        "auc": auc
    }
    
    return predicted_labels, true_labels, metrics



############## PPrediction using TTA #################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from typing import List, Tuple
from albumentations import (
HorizontalFlip,
ElasticTransform,
RandomBrightnessContrast,
RandomScale, Rotate
)


def horizontalFlip(image):
    image_aug = HorizontalFlip(p=1)
    augmented_image = image_aug(image=image)['image']
    
    return augmented_image

def randomBrightnessContrast(image):
    image_aug = RandomBrightnessContrast(p=1,brightness_limit=0.5, contrast_limit=0.4)
    augmented_image = image_aug(image=image)['image']
    
    return augmented_image

def elasticTransform(image):
    image_aug = ElasticTransform(p=1)
    augmented_image = image_aug(image=image)['image']
    
    return augmented_image


def randomScaling(image):
    image_aug = RandomScale(p=1, scale_limit = 0.2)
    augmented_image = image_aug(image=image)['image']
    
    return augmented_image

def rotate(image):
    image_aug = Rotate(p=1, limit = 45)
    augmented_image = image_aug(image=image)['image']
    
    return augmented_image


augmentation_funcs = [
    horizontalFlip, 
    randomBrightnessContrast,
    elasticTransform,
    randomScaling,
    rotate
]


def tta_predict(
    model: torch.nn.Module,
    image_path: str,
    augmentation_funcs: List[str],
    image_transform,
    device: torch.device
):
    
    
    original_image = Image.open(image_path)
    tta_predictions = []

    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        transformed_image = image_transform(original_image).unsqueeze(dim=0)
        original_pred = model(transformed_image.to(device))
        tta_predictions.append(torch.softmax(original_pred, dim=1))
    
    # Augmented image predictions
    for aug_func in augmentation_funcs:
        aug_image_np = aug_func(np.array(original_image))
        aug_image = Image.fromarray(aug_image_np)
        
        with torch.inference_mode():
            transformed_aug_image = image_transform(aug_image).unsqueeze(dim=0)
            aug_pred = model(transformed_aug_image.to(device))
            tta_predictions.append(torch.softmax(aug_pred, dim=1))
           
    # use average to get the predictions
    final_pred_probs = torch.mean(torch.stack(tta_predictions), dim=0)
        
    return final_pred_probs
          
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_skin_tone(model, dataset,transform, device):
        
    acc = 0
    all_true_labels = []
    all_pred_probs = []
    predicted_labels = []
    
    
    model.to(device)
    model.eval()
    
    
    for index, row in dataset.iterrows():
        
        image_path = row["DDI_path"]
        true_label = row["malignant"]
        
        
        final_pred_probs = tta_predict(model, image_path,augmentation_funcs, image_transform = transform, device=device)
        predicted_class = torch.argmax(final_pred_probs, dim=1).item()

        if predicted_class == true_label:
            acc += 1
            
            
        all_true_labels.append(true_label)
        all_pred_probs.append(final_pred_probs[0, 1].item())
        predicted_labels.append(predicted_class)
    
    accuracy = acc / len(dataset) if len(dataset) > 0 else 0
    auc_score = roc_auc_score(all_true_labels, np.array(all_pred_probs)) if all_true_labels else 0
    
    metrics = {"accuracy": accuracy, "auc": auc_score}
    
    
    return all_true_labels, predicted_labels, metrics
    
        
        