import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
import timm

################# Data setup ###########################s
class SkinDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["DDI_path"].replace("\\", "/")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['malignant'], dtype=torch.float).unsqueeze(0) 
        return image, label

############## Swin Transformer for features extraction ##################""
class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=1):
        super(SkinCancerModel, self).__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in list(self.backbone.layers[-2:].parameters()):
            param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  
        )
    
    def forward(self, image):
        image_features = self.backbone(image)
        output = self.classifier(image_features)
        return output.squeeze(1)  

############ train_model is our function that we'll use for training ################
def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).squeeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    
    val_auc, val_auc_pr = evaluate_model(model, val_loader, device)
    return total_loss / len(train_loader), auc, auc_pr, val_auc, val_auc_pr

############# evaluate_model is the function that help us to make evaluation of our model ##############
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device).squeeze(1)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.sigmoid(outputs).cpu().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    return auc, auc_pr

################ Get the data to apply our model on it ##################""
data_dir = "data"
train_csv = os.path.join(data_dir, "train_data.csv")
test_csv = os.path.join(data_dir, "test_data.csv")

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SkinDataset(train_csv, transform=transform)
test_dataset = SkinDataset(test_csv, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


### plit the train data into 80% train and 20% test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_auc, train_auc_pr, val_auc, val_auc_pr = train_model(model, train_loader, val_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | Train AUC={train_auc:.4f} | Train AUC-PR={train_auc_pr:.4f} | Val AUC={val_auc:.4f} | Val AUC-PR={val_auc_pr:.4f}")

# Save the model
torch.save(model.state_dict(), "image_only_skin_cancer.pth")
print("Model saved to image_only_skin_cancer.pth")
