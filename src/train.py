import torch
import torch.optim as optim
import torch.nn as nn
import os
from config import config
from backbone import get_backbone_model
from concept_util import load_concepts, get_concept_embeddings
from cbm import ConceptBottleneckLayer, SparseFinalLayer, ConceptBottleneckModel
from loader import get_train_loader

# Training loop
def train(net, trainloader, device):

    net.to(device)
    for epoch in range(config.NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            l1_penalty = config.L1_LAMBDA * torch.norm(sparse_layer.fc.weight, 1)
            loss = loss + l1_penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    torch.save(net.state_dict(), './models/cifar10_cbm.pth')
    print("Model saved successfully to './models/cifar10_cbm.pth'")

if __name__ == "__main__":
    # Load concepts and embeddings
    filtered_concepts = load_concepts()
    concept_embeddings = get_concept_embeddings(filtered_concepts)

    # Initialize model components
    backbone, feature_dim = get_backbone_model()
    concept_dim = len(concept_embeddings)
    num_classes = 10  # Set based on dataset

    cbl = ConceptBottleneckLayer(feature_dim, concept_dim)
    sparse_layer = SparseFinalLayer(concept_dim, num_classes)
    model = ConceptBottleneckModel(backbone, cbl, sparse_layer)

    # Training setup
    train_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Start training
    train(model, train_loader, device)