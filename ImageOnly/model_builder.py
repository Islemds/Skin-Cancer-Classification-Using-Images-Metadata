import torch
from torch import nn
from torchvision.models import (
    efficientnet_b5, EfficientNet_B5_Weights, 
    inception_v3, Inception_V3_Weights, 
    densenet121, DenseNet121_Weights
)

class ImageOnlyModel(nn.Module):
    def __init__(self, backbone="efficientnet_b5", num_classes=2):
        super(ImageOnlyModel, self).__init__()
        
        self.backbone_name = backbone.lower()
        
        if self.backbone_name == "efficientnet_b5":
            self.feature_extractor = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
            in_features = self.feature_extractor.classifier[1].in_features
            self.feature_extractor.classifier[1] = nn.Identity()  

        elif self.backbone_name == "inception_v3":
            self.feature_extractor = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            in_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()


        elif self.backbone_name == "densenet121":
            self.feature_extractor = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = self.feature_extractor.classifier.in_features
            self.feature_extractor.classifier = nn.Identity()  

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Freeze feature extractor layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.backbone_name == "inception_v3":
            features = self.feature_extractor(x)
            features = features.logits if hasattr(features, 'logits') else features
        else: 
            features = self.feature_extractor(x)
        
        output = self.classifier(features)
        return output
