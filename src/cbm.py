import torch
import torch.nn as nn

class ConceptBottleneckLayer(nn.Module):
    def __init__(self, feature_dim, concept_dim):
        super(ConceptBottleneckLayer, self).__init__()
        self.Wc = nn.Linear(feature_dim, concept_dim)
    
    def forward(self, features):
        return self.Wc(features)  # Projects features to concept space

class SparseFinalLayer(nn.Module):
    def __init__(self, concept_dim, num_classes):
        super(SparseFinalLayer, self).__init__()
        self.fc = nn.Linear(concept_dim, num_classes)
    
    def forward(self, concepts):
        return self.fc(concepts)  # Maps concepts to class predictions

class ConceptBottleneckModel(nn.Module):
    def __init__(self, backbone, cbl, sparse_layer):
        super(ConceptBottleneckModel, self).__init__()
        self.backbone = backbone
        self.cbl = cbl
        self.sparse_layer = sparse_layer
    
    def forward(self, x):
        features = self.backbone(x)
        concepts = self.cbl(features)
        output = self.sparse_layer(concepts)
        return output
