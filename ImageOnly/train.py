import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
from pathlib import Path
import torchvision
import pandas as pd
from utils import get_data_transform
import warnings

warnings.simplefilter("default")


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# hyperparameters
NUM_EPOCHS = 40
NUM_EPOCHS_CROSS_VALIDATION = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
N_SPLITS = 10
PATIENCE = 3
BACKBONE = "efficientnet_b5"

data_path = Path("data/")  

images_path = data_path
train_dir = images_path / "train"
test_dir = images_path / "test"

if not train_dir.exists():
    print(f"Train directory not found: {train_dir}")
if not test_dir.exists():
    print(f"Test directory not found: {test_dir}")
import os
print("Current working directory:", os.getcwd())

device = "cuda" if torch.cuda.is_available() else "cpu"

# data transform
data_transform = get_data_transform(backbone=BACKBONE)

train_data, test_data, num_classes = data_setup.create_datasets(train_dir, test_dir, data_transform)


best_model, results = engine.cross_validate_model(
    dataset=train_data, 
    backbone = BACKBONE,
    num_classes=num_classes, 
    epochs=NUM_EPOCHS_CROSS_VALIDATION, 
    batch_size=BATCH_SIZE, 
    learning_rate=LEARNING_RATE, 
    n_splits=N_SPLITS,
    device=device
)

results_df = pd.DataFrame(results)
results_df.to_csv(f"CV_{BACKBONE}_results_N_SPLITS={N_SPLITS}_EPOCHS={NUM_EPOCHS_CROSS_VALIDATION}_batch={BATCH_SIZE}_lr={LEARNING_RATE}.csv", index=False)
print(f"\Results saved")


best_model_full_data, metrics = engine.retrain_on_full_data(best_model= best_model, 
                            dataset = train_data,
                            num_classes=num_classes,  
                            epochs=NUM_EPOCHS, 
                            batch_size=BATCH_SIZE, 
                            learning_rate=LEARNING_RATE, 
                            patience=PATIENCE,
                            device=device)


target_dir = "models"
model_name = f"{BACKBONE}_N_Epochs={NUM_EPOCHS}_batch={BATCH_SIZE}_lr={LEARNING_RATE}.pth"


utils.save_model(model=best_model_full_data, target_dir=target_dir, model_name=model_name)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{BACKBONE}_results_N_Epochs={NUM_EPOCHS}_batch={BATCH_SIZE}_lr={LEARNING_RATE}.csv", index=False)
print(f"\nMetrics saved to")
