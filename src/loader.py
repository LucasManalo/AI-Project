from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config

def get_train_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = datasets.CIFAR10(root=config.DATA_DIR, train=True, transform=transform, download=True)
    return DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_data = datasets.CIFAR10(root=config.DATA_DIR, train=False, transform=transform, download=True)
    return DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)