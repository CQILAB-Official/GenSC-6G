import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, csv
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
from datetime import datetime
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.utils import save_image
from semantic_communication.noise import add_awgn_noise, add_rayleigh_noise

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='resnet50', help='[resnet50, dinov2, \
        vit]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--device', type=int, default=0, help='[0, 1]')
    parser.add_argument('--snr', type=int, default=30, help='Signal-to-noise ratio for dataset path')
    return parser.parse_args()


args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
logdir = f"logs/upsampling-featup/log-{args.model_name}/" if args.load_model is None else args.load_model
model_save_path = f'{logdir}/final_model.pth'

# Parse arguments
batch_size = args.batch_size
num_epochs = 51
train_snr = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model_name = args.model_name

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]  # Get the image file path
        return original_tuple + (path,)

train_folder = "GenSC-Testbed/GT_Images_Classification/Train"
test_folder = "GenSC-Testbed/GT_Images_Classification/Test"

train_dataset = ImageFolderWithPaths(root=train_folder, transform=transform)
test_dataset = ImageFolderWithPaths(root=test_folder, transform=transform)

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Decoder model
class Encoder(nn.Module):
    def __init__(self, model_name='resnet50'):
        super(Encoder, self).__init__()
        # Load the FeatUp model
        self.featup = torch.hub.load("mhamilton723/FeatUp", model_name, use_norm=False)

    def forward(self, x):
        x = self.featup.model.model(x)
        return x

model = Encoder().to(device)

# Print the model's number of parameters before running the test
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters.")

model.eval()

# Extract features and save as .npy files
all_features = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for images, labels, paths in tqdm(dataloader_train):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_filenames.extend(paths)

# Concatenate all features, labels, and filenames
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_filenames = np.array(all_filenames)

# Save features, labels, and filenames as .npy files
output_dir = "GenSC-Testbed/Extracted_Features_Upsampling/Train"
os.makedirs(output_dir, exist_ok=True)

features_path = os.path.join(output_dir, f"{args.model_name}_features.npy")
labels_path = os.path.join(output_dir, f"{args.model_name}_labels.npy")
filenames_path = os.path.join(output_dir, f"{args.model_name}_filenames.npy")

np.save(features_path, all_features)
np.save(labels_path, all_labels)
np.save(filenames_path, all_filenames)

print(f"Features saved to {features_path}")
print(f"Labels saved to {labels_path}")
print(f"Filenames saved to {filenames_path}")

# Print the len of all features
print(f"Length of all features: {len(all_features)}")
print(f"Length of all labels: {len(all_labels)}")
print(f"Length of all filenames: {len(all_filenames)}")


# Extract features and save as .npy files
all_features = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for images, labels, paths in tqdm(dataloader_test):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_filenames.extend(paths)

# Concatenate all features, labels, and filenames
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_filenames = np.array(all_filenames)

# Save features, labels, and filenames as .npy files
output_dir = "GenSC-Testbed/Extracted_Features_Upsampling/Test"
os.makedirs(output_dir, exist_ok=True)

features_path = os.path.join(output_dir, f"{args.model_name}_features.npy")
labels_path = os.path.join(output_dir, f"{args.model_name}_labels.npy")
filenames_path = os.path.join(output_dir, f"{args.model_name}_filenames.npy")

np.save(features_path, all_features)
np.save(labels_path, all_labels)
np.save(filenames_path, all_filenames)

print(f"Features saved to {features_path}")
print(f"Labels saved to {labels_path}")
print(f"Filenames saved to {filenames_path}")


# Print the len of all features
print(f"Length of all features: {len(all_features)}")
print(f"Length of all labels: {len(all_labels)}")
print(f"Length of all filenames: {len(all_filenames)}")