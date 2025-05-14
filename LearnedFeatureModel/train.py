#### Train.py ######
import os
from pathlib import Path
import torch

from torchvision import transforms
from pathlib import Path
import torchvision
import pandas as pd

from data_setup import create_datasets

import data_setup, engine, model_builder, utils

import warnings
warnings.simplefilter("default")

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# hyperparameters
NUM_EPOCHS = 40
NUM_EPOCHS_CROSS_VALIDATION = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
N_SPLITS = 10
num_classes = 2
PATIENCE = 3
FUSION_OPERATION = "mul" 

BASE_DIR = Path(__file__).resolve().parent  
images_path = BASE_DIR / "data"
train_images_path = images_path / "train"
test_images_path = images_path / "test"
train_metadata_path = images_path / "train_data.csv"
test_metadata_path = images_path / "test_data.csv"

def check_path_exists(path, path_type="file/directory"):
    if not path.exists():
        print(f"{path_type.capitalize()} not found: {path}")
        return False
    print(f"{path_type.capitalize()} exists: {path}")
    return True

print("Checking directories and metadata files...\n")
check_path_exists(train_images_path, "directory")
check_path_exists(test_images_path, "directory")
check_path_exists(train_metadata_path, "file")
check_path_exists(test_metadata_path, "file")

print("\nCurrent working directory:", Path.cwd())


# Load metadata
train_metadata = pd.read_csv(train_metadata_path)
test_metadata = pd.read_csv(test_metadata_path)
train_metadata.reset_index(drop=True, inplace=True)
test_metadata.reset_index(drop=True, inplace=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# data transform
weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
data_transform = weights.transforms()
data_transform

train_dataset = create_datasets(images_path=train_images_path, metadata=train_metadata, transform=data_transform)
test_dataset = create_datasets(images_path=test_images_path, metadata=test_metadata, transform=data_transform)

best_model, results = engine.cross_validate_model(
    dataset=train_dataset, 
    num_classes=num_classes, 
    epochs=NUM_EPOCHS_CROSS_VALIDATION, 
    batch_size=BATCH_SIZE, 
    learning_rate=LEARNING_RATE, 
    fusion_operation=FUSION_OPERATION,
    n_splits=N_SPLITS
)

results_df = pd.DataFrame(results)
results_df.to_csv(f"CV_{FUSION_OPERATION}_results_efficientnet_b5_N_SPLITS={N_SPLITS}_Epochs={NUM_EPOCHS_CROSS_VALIDATION}_batch={BATCH_SIZE}_lr={LEARNING_RATE}_melanoma.csv", index=False)
print(f"\Results saved")


best_model_full_data, metrics = engine.retrain_on_full_data(best_model= best_model, 
                            dataset = train_dataset,
                            num_classes=num_classes,  
                            epochs=NUM_EPOCHS, 
                            batch_size=BATCH_SIZE, 
                            learning_rate=LEARNING_RATE,
                            patience = PATIENCE,
                            device=device)

target_dir = "models"
model_name = f"{FUSION_OPERATION}_efficientnet_b5_N_Epochs={NUM_EPOCHS}_batch={BATCH_SIZE}_lr={LEARNING_RATE}_melanoma.pth"

# Save our model
utils.save_model(model=best_model_full_data, target_dir=target_dir, model_name=model_name)


# Save metrics 
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{FUSION_OPERATION}_results_efficientnet_b5_N_Epochs={NUM_EPOCHS}_batch={BATCH_SIZE}_lr={LEARNING_RATE}_melanoma.csv", index=False)
print(f"\nMetrics saved to")



