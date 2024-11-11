import torch
import matplotlib.pyplot as plt
from backbone import get_backbone_model  # Import function to load ResNet18 backbone
from concept_util import load_concepts, get_concept_names
from cbm import ConceptBottleneckLayer, SparseFinalLayer, ConceptBottleneckModel  # Import CBL and sparse layer
from loader import get_test_loader  # Import data loaders


def visualize_concept_activations(model, input_image, concept_names, threshold=0.5):
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        # Pass the input image through the model's backbone and Concept Bottleneck Layer
        features = model.backbone(input_image.unsqueeze(0))
        concept_activations = model.cbl(features).squeeze()  # Get concept activations and remove batch dimension

    # Filter and display activations above the threshold
    high_activations = [(concept_names[i], concept_activations[i].item())
                        for i in range(len(concept_activations))
                        if concept_activations[i].item() > threshold]

    # Sort by activation strength (highest first)
    high_activations.sort(key=lambda x: x[1], reverse=True)

    print("Top concept activations for the input image:")
    for concept, activation in high_activations:
        print(f"{concept}: {activation:.4f}")

    # Plotting the activations
    concepts, activations = zip(*high_activations)  # Unzip for plotting
    plt.figure(figsize=(6, 6))
    plt.barh(concepts, activations, color="skyblue")
    plt.xlabel("Activation Level")
    plt.title("Top Concept Activations")
    plt.gca().invert_yaxis()  # Highest activations at the top
    for index, activation in enumerate(activations):
        plt.text(activation, index, f"{activation:.2f}", va='center')  # Add text to the right of the bar
    plt.show()

# Evaluate the CBM model
def evaluate_model(net, testloader, device):
    net.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient calculation as it's not needed for evaluation
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Move data to GPU if available
            outputs = net(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class with the highest score
            total += labels.size(0)  # Total number of test samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    # Set up the device: use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    # Load both the training and test data using the load_cifar10 function
    testloader = get_test_loader()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Initialize the CBM model components
    backbone, feature_dim = get_backbone_model()  # Load backbone (ResNet18)
    concept_dim = 157  # Set this according to your filtered concept count
    num_classes = 10   # For CIFAR-10 dataset, adjust if using another dataset

    # Create Concept Bottleneck Layer and Sparse Final Layer
    cbl = ConceptBottleneckLayer(feature_dim, concept_dim)
    sparse_layer = SparseFinalLayer(concept_dim, num_classes)

    # Combine into the CBM model
    cbm_model = ConceptBottleneckModel(backbone, cbl, sparse_layer)
    cbm_model.to(device)

    # Load the saved CBM model weights (modify this path as per where you saved the model)
    model_path = './models/cifar10_cbm.pth'
    cbm_model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate the CBM model
    # evaluate_model(cbm_model, testloader, device)

    input_image = images[0].to(device)
    filtered_concepts = load_concepts()
    concept_names = get_concept_names(filtered_concepts)
    visualize_concept_activations(cbm_model, input_image, concept_names, threshold=0.5)
