import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# configurable MLP class
class FNN(nn.Module):
    """
    input_dim: dimensions of the input data
    hidden_dim: number of neurons per hidden layer
    num_classes: number of classes to be predicted
    num_hidden_layers: number of hidden layers in the FNN
    dropout: rate for neuron dropout, default to 0 (no dropout)
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden_layers, dropout=0.0):
        super(FNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.flatten = nn.Flatten()

        self.m1_stack = nn.Sequential(
            # H1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            # Output Layer
            nn.Linear(hidden_dim, num_classes)
        )

        self.m2_stack = nn.Sequential(
            #H1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            #H2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #H3
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #Output Layer
            nn.Linear(hidden_dim, num_classes)
        )

        self.m3_stack = nn.Sequential(
            #H1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            #H2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #H3
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #H4
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #H5
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #Output Layer
            nn.Linear(hidden_dim, num_classes)
        )

        self.m4_stack = nn.Sequential(
            #H1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            #H2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            #H3
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            #H4
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            #H5
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            #Output Layer
            nn.Linear(hidden_dim, num_classes)
        )
    
    # forward pass of the model; outputs logits
    # model configuration varys between 4 stacks depending on model parameters
    def forward(self, x):
        # Convert multidimensional vector into flat value for processing
        x = self.flatten(x)

        # Determine model stack to use based on number of hidden layers and dropout
        if (self.num_hidden_layers == 1):
            logits = self.m1_stack(x)
        elif (self.num_hidden_layers == 3):
            logits = self.m2_stack(x)
        elif (self.num_hidden_layers == 5 and self.dropout == 0.0):
            logits = self.m3_stack(x)
        elif (self.num_hidden_layers == 5 and self.dropout > 0.0):
            logits = self.m4_stack(x)
        
        return logits

# Customer Dataset for feature vectors
class CustomDataset(Dataset):
    def __init__(self, n_samples=10000, n_features=256, n_classes=5):
        # Generate training features
        self.X = torch.randn(n_samples, n_features)
        # Generate target labels
        self.y = torch.randint(0, n_classes, (n_samples,))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Train model on train data with a forward and backwards pass
# Validate model on val data with only forward pass
# returns dictonary containing accuracy and loss for train and val data
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs):
    # dict to track train and validation loss and accuracy across epochs
    report_metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        # Set model to train mode
        model.train()

        # tracking variables
        current_loss = 0
        correct = 0
        total = 0

        # Train on train data
        for inputs, labels in train_loader:
            # reset gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # Compute loss as predicted vs. actual based on parameter loss function
            loss = loss_fn(outputs, labels)

            # Backwards pass to compute gradients
            loss.backward()
            # Update model parameters based on gradient
            optimizer.step()

            # track loss and accuracy
            #NOTE: Use .item() to get gradient as float
            #NOTE: input.size(0) is batch size; current loss = avg_gradient_value * number of samples
            current_loss += loss.item() * inputs.size(0)

            #NOTE: gets value (discarded) & index of class with highest prediction probability;
            #       param 1 indicates we want max class of current sample, rather than 0 for max of entire batch
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # calculate training loss and accuracy
        train_loss = current_loss / total
        train_acc = correct / total

        # Evaluate on validation data
        model.eval()

        val_current_loss = 0
        val_correct = 0
        val_total = 0

        #NOTE: No backward pass as we are validating; no grad disables gradient calculation
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Forward pass
                outputs = model(inputs)
                # Compute loss
                loss = loss_fn(outputs, labels)

                # Compute metrics
                val_current_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # calculate validation loss and accuracy
        val_loss = val_current_loss / val_total
        val_acc = val_correct / val_total

        report_metrics["train_loss"].append(train_loss)
        report_metrics["train_accuracy"].append(train_acc)
        report_metrics["val_loss"].append(val_loss)
        report_metrics["val_accuracy"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    return report_metrics

# Evaluate trained model performance with a single backpass
# returns dict containg test accuracy and loss
def test_model(model, test_loader, loss_fn):
    # dict for tracking test loss and accuracy
    report_metrics = {
        "test_loss": [],
        "test_accuracy": []
    }

    # set model to evaluate mode
    model.eval()

    # tracking variables
    current_loss = 0
    correct = 0
    total = 0

    # no backpass, so no gradient
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = loss_fn(outputs, labels)

            # compute metrics
            current_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # calculate test loss and accuracy
    test_loss = current_loss / total
    test_acc = correct / total

    report_metrics["test_loss"].append(test_loss)
    report_metrics["test_accuracy"].append(test_acc)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | ")

    return report_metrics

def extract_confidence(model, test_loader):
    confidence_scores = []
    true_labels = []

    # set model to evaluate mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass to compute logits
            outputs = model(inputs)

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Extract maximum class probability for each sample
            scores, _ = torch.max(probabilities, dim=1)

            # Convert scores and labels from tensor to numpy array and save to list
            confidence_scores.extend(scores.numpy())
            true_labels.extend(labels.numpy())
    return confidence_scores, true_labels

def mean_confidence_per_group(confidence_scores, true_labels):
    per_group_confidence_scores = []

    for class_id in range(5):
        # group scores which have the same class ID
        group_scores = [confidence_scores[i] for i in range(len(true_labels)) if true_labels[i] == class_id]
        per_group_confidence_scores.append( round(sum(group_scores) / len(group_scores), 2) )
    return per_group_confidence_scores

# Helper function for generating a learning curve  of accuracy & loss for both train and validation data
def plot_learning_curve(metrics):
    # create epoch range based on count of metrics, +1 so epochs are not 0-indexed
    epochs = range(1, len(metrics["train_accuracy"])+1)

    # create figure with 2 subplots side by side (1 row, 2 cols)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # plot accuracy in column 1
    # plot training accuracy over epoch with blue line
    ax1.plot(epochs, metrics['train_accuracy'], 'b-', label='Training Accuracy')
    #  plot validation accuracy over epoch with red line
    ax1.plot(epochs, metrics['val_accuracy'], 'r-', label='Validation Accuracy')
    # Set graph x and y axis labels
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Epochs')
    ax1.legend()
    ax1.grid(True)

    # plot loss in column 2
    # plot training loss over epoch with green line
    ax2.plot(epochs, metrics['train_loss'], 'g-', label='Training Loss')
    # plot validation accuracy over epoch with yellow line
    ax2.plot(epochs, metrics['val_loss'], 'y-', label='Validation Loss')
    # Set graph x and y axis labels
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Loss over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Helper to plot confidence histogram
def plot_confidence_histogram(scores, model_name):
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black', range=(0, 1))

    # Adding labels and title
    plt.xlabel('Confidence')
    plt.ylabel('# of Samples')
    plt.title('Sample Confidence of ' + model_name)

    # Display the plot
    plt.show()

# Helper function for saving model results in tabular format
def generate_table(train_metrics, model_name):
    # non-0-indexed epoch count
    epochs = list(range(1,len(train_metrics["train_accuracy"])+1))

    # create dataframe using metric values and model name
    # metrics rounded to 3f for readablility
    df = pd.DataFrame({
        'epoch': epochs,
        model_name + '_train_loss': train_metrics["train_loss"],
        model_name + '_val_loss': train_metrics["val_loss"],
        model_name + '_train_accuracy': train_metrics["train_accuracy"],
        model_name + '_val_accuracy': train_metrics["val_accuracy"],
        })
    
    df = df.round(3)
    
    return df

def main():
    print("Lab 3 Deep Learning & Neural Networks")

    # Set global random seed
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Set parameters
    class_count = 5
    neuron_width = 256
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    sample_size = 10000

    # generate sample data
    data = CustomDataset(sample_size, neuron_width, class_count)

    # split into train, test and validation data sets
    # Train size 3 (for 5 classes)
    train_size = int(0.7 * len(data))
    # Val size 1 (for 5 classes)
    val_size = int(0.15 * len(data))
    # test size 1 (for 5 classes)
    test_size = len(data) - train_size - val_size

    train, val, test = random_split(data, [train_size, val_size, test_size])

    # create dataloaders
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    # Use cross entropy for loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize models and optimizers 
    m1 = FNN(neuron_width, neuron_width, class_count, 1, 0.0)
    m1_opt = optim.Adam(m1.parameters(), lr=learning_rate)

    m2 = FNN(neuron_width, neuron_width, class_count, 3, 0.0)
    m2_opt = optim.Adam(m2.parameters(), lr=learning_rate)

    m3 = FNN(neuron_width, neuron_width, class_count, 5, 0.0)
    m3_opt = optim.Adam(m3.parameters(), lr=learning_rate)

    m4 = FNN(neuron_width, neuron_width, class_count, 5, 0.4)
    m4_opt = optim.Adam(m4.parameters(), lr=learning_rate)

    # Train models on train, val, and test datasets
    # print("========== M1 Training ==========")
    # m1_res = train_model(m1, train_dl, val_dl, m1_opt, loss_fn, epochs)
    # m1_test = test_model(m1, test_dl, loss_fn)
    # m1_conf, m1_conf_labels = extract_confidence(m1, test_dl)

    print("========== M2 Training ==========")
    m2_res = train_model(m2, train_dl, val_dl, m2_opt, loss_fn, epochs)
    m2_test = test_model(m2, test_dl, loss_fn)
    m2_conf, m2_conf_labels = extract_confidence(m2, test_dl)

    # print("========== M3 Training ==========")
    # m3_res = train_model(m3, train_dl, val_dl, m3_opt, loss_fn, epochs)
    # m3_test = test_model(m3, test_dl, loss_fn)
    # m3_conf, m3_conf_labels = extract_confidence(m3, test_dl)

    print("========== M4 Training ==========")
    m4_res = train_model(m4, train_dl, val_dl, m4_opt, loss_fn, epochs)
    m4_test = test_model(m4, test_dl, loss_fn)
    m4_conf, m4_conf_labels = extract_confidence(m4, test_dl)

    # Plot confidence histograms
    # plot_confidence_histogram(m1_conf, "M1")
    plot_confidence_histogram(m2_conf, "M2")
    # plot_confidence_histogram(m3_conf, "M3")
    plot_confidence_histogram(m4_conf, "M4")

    # Display group mean confidence
    m2_scores = mean_confidence_per_group(m2_conf, m2_conf_labels)
    m4_scores = mean_confidence_per_group(m4_conf, m4_conf_labels)

    print("===== M2 Confidence =====")
    for i, score in enumerate(m2_scores):
        print(f"Class {i} Confidence: ", round(float(score) * 100, 2), "%")

    print("===== M4 Confidence =====")
    for i, score in enumerate(m4_scores):
        print(f"Class {i} Confidence: ", round(float(score) * 100, 2), "%")
    
if __name__ == "__main__":
    main()