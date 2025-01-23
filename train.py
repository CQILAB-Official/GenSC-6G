import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import sys
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score, roc_curve, auc,
                             recall_score, precision_score, confusion_matrix, 
                             precision_recall_curve, average_precision_score)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='[inceptionv3, resnet, \
        qresnet, vit, swin, mobilenet, efficientnet, vgg, qcnn]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--device', type=int, default=0, help='[0, 1]')
    parser.add_argument('--snr', type=int, default=10, help='Signal-to-noise ratio for dataset path')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Setup the quantum device and the number of qubits
n_qubits = 10
dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

weight_shapes = {"weights": (10, n_qubits)}

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    def forward(self, x):
        return self.qlayer(x)

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(x)
    

class DecoderQuantum(nn.Module):
    def __init__(self, num_classes):
        super(DecoderQuantum, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, n_qubits),
        )
        self.qnet = QNet()
        self.fc2 = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.qnet(x)
        return self.fc2(x)

model_parsed = args.model_name
if model_parsed in ['qresnet', 'qvit', 'qnn']:
    model_parsed = model_parsed[1:]

# Load the features, labels, and filenames
train_feature_path = f"GenSC-Testbed/AWGN_Generated/Train/{args.snr}/{model_parsed}_features.npy"
train_label_path = f"GenSC-Testbed/AWGN_Generated/Train/{args.snr}/{model_parsed}_labels.npy"
train_features = np.load(train_feature_path)
train_labels = np.load(train_label_path)
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

test_feature_path = f"GenSC-Testbed/AWGN_Generated/Test/{args.snr}/{model_parsed}_features.npy"
test_label_path = f"GenSC-Testbed/AWGN_Generated/Test/{args.snr}/{model_parsed}_labels.npy"
test_features = np.load(test_feature_path)
test_labels = np.load(test_label_path)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Set up the device, model, loss function, and optimizer
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
num_classes = len(torch.unique(train_labels))  # Number of classes

if args.model_name in ['qresnet', 'qvit', 'qcnn']:
    model = DecoderQuantum(num_classes=num_classes).to(device)
else:
    model = Decoder(num_classes=num_classes).to(device)

# Print the model's number of parameters before running the test
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters.")

# Exit / stop python
sys.exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def save_confusion_matrix_csv(cm, epoch):
    """ Save the confusion matrix to a CSV file. """
    cm_df = pd.DataFrame(cm)
    full_filename = f"logs/{args.snr}/{args.model_name}/cm_epoch{epoch}.csv"
    # Make folder if not exist
    os.makedirs(os.path.dirname(full_filename), exist_ok=True)
    cm_df.to_csv(full_filename, index=False)


# Training function
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in tqdm(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Train - Epoch: {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


# Testing function
def test_model(model, dataloader, criterion, epoch):
    model.eval()
    test_loss = 0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    test_loss /= len(dataloader)
    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average='macro')
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')
    cm = confusion_matrix(ground_truths, predictions)
    save_confusion_matrix_csv(cm, epoch) 
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    # Save to txt file
    with open(f"logs/{args.snr}/{args.model_name}/test_results_{epoch}.txt", "a") as f:
        f.write(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

# Training loop
num_epochs = 150
best_acc = 0.0

for epoch in range(1, num_epochs + 1):
    train_model(model, train_loader, criterion, optimizer, epoch)
    
    # Test the model every 10 epochs
    if epoch % 10 == 0:
        acc = test_model(model, test_loader, criterion, epoch)

print(f"Best test accuracy: {best_acc:.2f}%")
      