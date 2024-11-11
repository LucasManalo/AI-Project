import torch
from torchvision import models

def get_backbone_model():
    # Load the ResNet-18 backbone
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Retrieve the feature dimension from the original final layer
    feature_dim = backbone.fc.in_features
    
    # Replace the final classification layer with an identity layer
    backbone.fc = torch.nn.Identity()

    return backbone, feature_dim
