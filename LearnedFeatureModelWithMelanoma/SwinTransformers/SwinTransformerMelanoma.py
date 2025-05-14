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

################# Data setup ###########################
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
        
        # Normalize metadata
        metadata = torch.tensor([
            row['skin_tone_12'],
            row['skin_tone_34'],
            row['skin_tone_56'],
            row['Disease_Group_Non_melanoma'],
            row['Disease_Group_melanoma']
        ], dtype=torch.float)
        metadata = (metadata - metadata.mean()) / (metadata.std() + 1e-6)
        
        label = torch.tensor(row['malignant'], dtype=torch.float).unsqueeze(0)
        return image, metadata, label

############## Swin Transformer for extract features from images ##################""
class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=1):
        super(SkinCancerModel, self).__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in list(self.backbone.layers[-2:].parameters()):  
            param.requires_grad = True
        
        self.image_fc = nn.Linear(feature_dim, 512)
        self.metadata_fc = nn.Linear(5, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 1) 
        )
    
    def forward(self, image, metadata):
        image_features = self.image_fc(self.backbone(image))
        metadata_features = self.metadata_fc(metadata)
        fused_features = torch.cat((image_features, metadata_features), dim=1)
        output = self.classifier(fused_features)
        return output.squeeze(1) 


############ train_model is our function that we'll use for training ################
def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    for images, metadata, labels in train_loader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device).squeeze(1) 
        
        optimizer.zero_grad()
        outputs = model(images, metadata) 
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
        for images, metadata, labels in data_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device).squeeze(1)
            outputs = model(images, metadata)
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


### plit the train data into 80% train and 20% test
train_dataset = SkinDataset(train_csv, transform=transform)
test_dataset = SkinDataset(test_csv, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



##### Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerModel().to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Train Model with validation monitoring
num_epochs = 10
for epoch in range(num_epochs):
    print("dh")
    train_loss, train_auc, train_auc_pr, val_auc, val_auc_pr = train_model(model, train_loader, val_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | Train AUC={train_auc:.4f} | Train AUC-PR={train_auc_pr:.4f} | Val AUC={val_auc:.4f} | Val AUC-PR={val_auc_pr:.4f}")

# Save the model
torch.save(model.state_dict(), f"skin_cancer_melanoma_epochs={num_epochs}.pth")
print(f"Model saved to skin_cancer_melanoma_epochs={num_epochs}.pth")
