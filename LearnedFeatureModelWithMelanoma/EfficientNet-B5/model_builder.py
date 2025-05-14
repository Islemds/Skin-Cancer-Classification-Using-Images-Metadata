import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights


class LearnedFeatureFusionModel(nn.Module):
    def __init__(self, num_classes=2, fusion_operation="mul"):
        super(LearnedFeatureFusionModel, self).__init__()
        self.fusion_operation = fusion_operation
        
        self.feature_extractor = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        
        # Freeze EfficientNet feature extractor weights
        for param in self.feature_extractor.features.parameters():
            param.requires_grad = False
        
        in_features = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier[1] = nn.Identity()
        
        self.image_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU()
        )
        
        self.skin_fc = nn.Sequential(
            nn.Linear(5, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        
        if fusion_operation == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(1024, 1024), 
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
    
    def forward(self, image, metadata):
        image_features = self.feature_extractor(image)
        image_features = self.image_fc(image_features)
        
        skin_features = self.skin_fc(metadata)
        
        if self.fusion_operation == "mul":
            fused_features = torch.mul(image_features, skin_features)
        elif self.fusion_operation == "add":
            fused_features = image_features + skin_features
        elif self.fusion_operation == "concat":
            fused_features = torch.cat((image_features, skin_features), dim=1)
        else:
            raise ValueError("Invalid fusion operation. Choose from ['mul', 'add', 'concat'].")
        
        # Classification
        out = self.classifier(fused_features)
        return out


