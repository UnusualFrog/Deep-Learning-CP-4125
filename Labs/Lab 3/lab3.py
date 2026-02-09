import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib


# Set hyperparameters
color_channels = 3 # 3 color channels for RGB
image_side_length  = 64 # images are size 64 squares
input_size = image_side_length * image_side_length # 64x64 pixel images
hidden_size = 100 # number of nodes in hidden layer
num_classes = 5 # number of classes, 0-4
num_epochs = 2 # number of times to loop through the entire dataset
batch_size = 100 # number of samples in one pass
learning_rate = 0.001 # learning rate


#A1.1) Set a static global random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.maunal_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Define transform to convert PIL image to Tensor
transform = transforms.ToTensor()

#A1.2) Build the dataset and split it
data = torchvision.datasets.FakeData(100, (color_channels, input_size, input_size), num_classes, transform=transform)

# split data 70/15/15
train_data, val_data, test_data = random_split(data, [.7, .15, .15])

#A1.3) Create Dataloaders (train/val/test)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#A2) Model Family
"""
@num_hl - number of hidden layers in NN, should be 1, 3 or 5
@hidden_dim - number of nodes in each hidden layer
@input_dim - size of input images
@num_classes - number of classes to be output by the final layer of the model
@dropout - percent of nodes to be dropped in each hidden layer to reduce overfitting, defaults to 0, should be 0.4 otherwise
"""
class FNN(nn.Module):
    def __init__(self, num_hl, hidden_dim, input_dim, num_classes, dropout=0.0):
        super(FNN,self).__init__()
        
        # Define function for flattening tensors to 1D 
        self.flatten = nn.Flatten()
        # List to hold variable-amount of hidden layers 1/3/5
        layers = []
        
        # First hidden layer
        # Convert input features to dimensions of hidden layers
        layers.append(nn.Linear(input_dim, hidden_dim))
        # Apply ReLU activation function to set values <0 to 0, capturing non-linear patterns
        layers.append(nn.ReLU())
        
        # Apply dropout to first layer if parameter > 0
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Additional hidden layers (for num_hl = 3 or 5)
        for _ in range(num_hl - 1):
            # transfer neurons from previous layer to current layer
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Apply ReLU activation function to current layer
            layers.append(nn.ReLU())
            
            # Apply dropout to current layer if >0
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        # Register all previous layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Convert multidimensional input images to 1d 
        x = self.flatten(x)
        
        # Compute logits by applying all layers of network to flattened input
        logits = self.network(x)
        
        return logits
    
M1 = FNN( 1, hidden_size, input_size * color_channels, num_classes, 0.0)