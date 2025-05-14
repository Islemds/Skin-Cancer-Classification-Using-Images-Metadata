import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

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

        metadata = torch.tensor([
            row['skin_tone_12'],
            row['skin_tone_34'],
            row['skin_tone_56'],
            row['Disease_Group_Non_melanoma'],
            row['Disease_Group_melanoma']
        ], dtype=torch.float)
        
        # Label
        label = torch.tensor(row['malignant'], dtype=torch.float)
        return image, metadata, label

################### Feature-wise Linear Modulation (FiLM) 
class FiLM(nn.Module):
    def __init__(self, metadata_dim, feature_dim):
        super(FiLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim * 2) 
        )
    
    def forward(self, features, metadata):
        gamma_beta = self.fc(metadata)  
        gamma, beta = gamma_beta.chunk(2, dim=1)  
        return features * gamma + beta

############# Build our model ##################
class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=1):
        super(SkinCancerModel, self).__init__()
        self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        for param in self.backbone.features.parameters():
            param.requires_grad = False  
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  
        
        self.film = FiLM(metadata_dim=5, feature_dim=in_features)  
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, image, metadata):
        image_features = self.backbone(image)  
        modulated_features = self.film(image_features, metadata)  
        output = self.classifier(modulated_features)
        return output


############ train_model is our function that we'll use for training ################
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    
    for images, metadata, labels in train_loader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, metadata).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.detach().cpu().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    return total_loss / len(train_loader), auc, auc_pr

# save our model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

######## predict function to make prediction on test set ###########
def predict(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images, metadata = images.to(device), metadata.to(device)
            outputs = model(images, metadata).squeeze()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    return auc, auc_pr

############### Get the data, transform it and split it into train and validation #############
data_dir = "data"
train_csv = os.path.join(data_dir, "train_data.csv")
test_csv = os.path.join(data_dir, "test_data.csv")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SkinDataset(train_csv, transform=transform)
test_dataset = SkinDataset(test_csv, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



##### Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 15
for epoch in range(num_epochs):
    train_loss, train_auc, train_auc_pr = train_model(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | AUC={train_auc:.4f} | AUC-PR={train_auc_pr:.4f}")

# Save the trained model
save_model(model, f"skin_cancer_model_melanoma_epochs={num_epochs}.pth")


test_auc, test_auc_pr = predict(model, test_loader, device)
print(f"Test AUC: {test_auc:.4f} | Test AUC-PR: {test_auc_pr:.4f}")
