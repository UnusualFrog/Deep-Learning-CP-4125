import torchvision, torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# FNN Class
class FNN(nn.Module):
    def __init__(self, num_h, hidden_dim, dropout,  ):
        super(FNN, self).__init__()
        
        # First hidden layer, use 3 channel (RGB) 64x64 pixel image to produce 512 features
        self.fc1 = nn.Linear(3 * num_h, hidden_dim)
        # Second hidden layer uses 512 features from first layer to produce 5 output classes
        self.fc2 = nn.Linear(512, 5)
        
    def forward(self, x):
        # Flatten 2D images into 1D vectors
        # -1 indicates infer the shape from batch_size
        x = x.view(-1, 3* 64 * 64)
        
        # Pass flattened input through first layer
        # Linear conversion of 784 input features to 512 specified features
        # Apply RELU to 512 features
        x = F.relu(self.fc1(x))
        
        # Pass first layer output through second layer
        # Linear transformation of 512 input features to 5 specified features
        x = self.fc2(x)
        
        return x
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        if (i + 1) % 200 == 0:
            print(
                f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
            running_loss = 0.0
            running_acc = 0.0


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += accuracy(outputs, labels)
    print(
        f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}')

def main():
    global_random_seed = torch.seed()

    # Define transform to convert PIL image to Tensor
    transform = transforms.ToTensor()
    
    # Set hyperparameters
    input_size = 64
    hidden_size = 256
    num_classes = 5
    num_epochs = 10
    batch_size = 20
    learning_rate = 0.01 
    

    # Generate dataset
    data = torchvision.datasets.FakeData(100, (3, input_size, input_size), num_classes, transform=transform)
    # print(data)

    # Split data into train, test and validation sets
    train_data, test_data, val_data = random_split(data, [0.6, 0.2, 0.2])
    # print(len(train_data))
    # print(len(test_data))
    # print(len(val_data))

      

    # Create dataloaders in batch sizes of 20 images    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Display batch shapes and labels
    # for batch_images, batch_labels in train_dataloader:
    #     print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")

    # Create model and set device, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M1 = FNN(input_size, hidden_size, 0, num_classes).to(device)
    # M2 = FNN(3, 1, 0).to(device)
    # M3 = FNN(5, 1, 0).to(device)
    # M4 = FNN(5, 1, 0.4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(M1.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train(M1, device, train_dataloader, criterion, optimizer, epoch)
        test(M1, device, test_dataloader, criterion)

if __name__ == '__main__':
    main()