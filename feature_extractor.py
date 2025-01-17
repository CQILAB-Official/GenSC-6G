import torch
import torch.nn as nn
import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from allmodels import InceptionEncoder, ResNetEncoder, ViTEncoder, SwinEncoder, MobileNetEncoder, EfficientNetEncoder, VGGEncoder, QCNNEncoder, CNNEncoder
import argparse
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys

# Pass through params
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='[inceptionv3, resnet, \
        qresnet, vit, swin, mobilenet, efficientnet, vgg, qcnn]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--device', type=int, default=0, help='[0, 1]')
    return parser.parse_args()

# Parse arguments
args = parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
root_folder = "GenSC-Testbed/GT_Images_Classification/Test"
transform = transforms.Compose([
    transforms.Resize((299, 299)) if args.model_name in ['inceptionv3', 'qinceptionv3'] else transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
batch_size = args.batch_size

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]  # Get the image file path
        return original_tuple + (path,)

train_dataset = ImageFolderWithPaths(root=root_folder, transform=transform)
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(train_dataset.classes)

if args.model_name == 'inceptionv3':
    model = InceptionEncoder().to(device)
elif args.model_name == 'resnet':
    model = ResNetEncoder().to(device)
elif args.model_name == 'qresnet':
    model = ResNetEncoder().to(device)
elif args.model_name == 'vit':
    model = ViTEncoder().to(device)
elif args.model_name == 'swin':
    model = SwinEncoder().to(device)
elif args.model_name == 'mobilenet':
    model = MobileNetEncoder().to(device)
elif args.model_name == 'efficientnet':
    model = EfficientNetEncoder().to(device)
elif args.model_name == 'vgg':
    model = VGGEncoder().to(device)
elif args.model_name == 'qcnn':
    model = CNNEncoder().to(device)


# Print the model's number of parameters before running the test
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters.")

# Exit / stop python
# sys.exit()

# Set model to evaluation mode
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
output_dir = "GenSC-Testbed/Extracted_Features_Classification/Sample"
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