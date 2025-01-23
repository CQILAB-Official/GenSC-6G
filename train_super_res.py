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
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='[resnet, vit]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--device', type=int, default=0, help='[0, 1]')
    parser.add_argument('--snr', type=int, default=30, help='Signal-to-noise ratio for dataset path')
    return parser.parse_args()


args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
logdir = f"logs/upsampling/log-{args.model_name}/" if args.load_model is None else args.load_model
model_save_path = f'{logdir}/final_model.pth'

# Parse arguments
batch_size = args.batch_size
num_epochs = 51

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Convolutional layers for downsampling and reducing the number of channels
        self.featup = torch.hub.load("mhamilton723/FeatUp", 'resnet50', use_norm=True)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # Reduce channels from 2048 to 1024
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=3, padding=1),   # Reduce channels from 1024 to 512
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),    # Reduce channels from 512 to 256
        )


    def forward(self, x):
        x = self.featup(x)
        print(x.shape)
        # Pass through convolution layers to reduce channels
        x = self.conv_layers(x)
        print(x.shape)
        return x

class DecoderViT(nn.Module):
    def __init__(self):
        super(DecoderViT, self).__init__()
        
        # Convolutional layers for downsampling and reducing the number of channels
        self.featup = torch.hub.load("mhamilton723/FeatUp", 'vit', use_norm=True)
        self.conv_layers = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(384, 64, kernel_size=3, padding=1),  # Reduce channels from 2048 to 1024
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),    # Reduce channels from 512 to 256
        )


    def forward(self, x):
        x = self.featup(x)
        # Pass through convolution layers to reduce channels
        x = self.conv_layers(x)

        return x

# Load the features, labels, and filenames
train_feature_path = f"GenSC-Testbed/AWGN_Generated/Train/{args.snr}/{args.model_name}_features.npy"
train_label_path = f"GenSC-Testbed/AWGN_Generated/Train/{args.snr}/{args.model_name}_labels.npy"
train_features = np.load(train_feature_path)
train_labels = np.load(train_label_path)
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

test_feature_path = f"GenSC-Testbed/AWGN_Generated/Test/{args.snr}/{args.model_name}_features.npy"
test_label_path = f"GenSC-Testbed/AWGN_Generated/Test/{args.snr}/{args.model_name}_labels.npy"
test_features = np.load(test_feature_path)
test_labels = np.load(test_label_path)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_image_path = f"GenSC-Testbed/AWGN_Generated/Train/{args.snr}/{args.model_name}_filenames.npy"
test_image_path = f"GenSC-Testbed/AWGN_Generated/Test/{args.snr}/{args.model_name}_filenames.npy"
train_image_filenames = np.load(train_image_path)
test_image_filenames = np.load(test_image_path)

print(len(train_features), len(train_labels), len(train_image_filenames))

transform = transforms.Compose([
    transforms.Resize((299, 299)) if args.model_name in ['inceptionv3', 'qinceptionv3'] else transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_input = transforms.Compose([
    transforms.Resize((299, 299)) if args.model_name in ['inceptionv3', 'qinceptionv3'] else transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_paths(image_filenames, transform=transform):
    image_tensors = []
    for img_path in image_filenames:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image)
        image_tensors.append(image_tensor)
    return torch.stack(image_tensors)


# Load images based on file paths
train_image_tensor = load_images_from_paths(train_image_filenames)
test_image_tensor = load_images_from_paths(test_image_filenames)

train_image_tensor_input = load_images_from_paths(train_image_filenames, transform_input)
test_image_tensor_input = load_images_from_paths(test_image_filenames, transform_input)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_features, train_labels, train_image_tensor, train_image_tensor_input)
test_dataset = TensorDataset(test_features, test_labels, test_image_tensor, test_image_tensor_input)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Set up the device, model, loss function, and optimizer
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
num_classes = len(torch.unique(train_labels))  # Number of classes

if args.model_name == 'resnet':
    model = Decoder().to(device)
elif args.model_name == 'vit':
    model = DecoderViT().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=logdir)
to_pil = transforms.ToPILImage()

input_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
])
def train(model, dataloader):
    model.train()  # Set the model to training mode
    train_loss = 0
    for features, labels, inputs, input_resized in tqdm(dataloader, desc="Training"):
        features, labels, inputs, input_resized = features.to(device), labels.to(device), inputs.to(device), input_resized.to(device)
        optimizer.zero_grad()
        x = add_awgn_noise(inputs, args.snr)
        outputs = model(x)
        
        # Transform the input to 224x224 using torchvision transforms
        loss = criterion(outputs, input_resized)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)
    
    print(f"Train Loss: {train_loss:.4f}")
    writer.add_scalar('Loss/train', train_loss, epoch)


from semantic_communication.metrics import calculate_psnr, calculate_ssim, calculate_lpips_similarity_tensor, calculate_lpips_similarity
def test(model, dataloader, epoch, test_snr=args.snr, is_psnr=False):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    psnr, lpips, ssim = 0, 0, 0
    
    epoch_dir = os.path.join(logdir, str(epoch))
    
    os.makedirs(epoch_dir, exist_ok=True)
    input_dir = os.path.join(epoch_dir, 'input')
    output_dir = os.path.join(epoch_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Open CSV file to save metrics
    csv_file_path = os.path.join(epoch_dir, 'metrics.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV header
        csv_writer.writerow(['Image_Index', 'PSNR', 'LPIPS', 'SSIM'])
        
        idx = 0
        with torch.no_grad():
            for features, labels, inputs, input_resized in tqdm(dataloader, desc="Evaluating"):
                features, labels, inputs, input_resized = features.to(device), labels.to(device), inputs.to(device), input_resized.to(device)

                x = add_awgn_noise(inputs, test_snr)
                outputs = model(x)
                inputs = input_resized
                loss = criterion(outputs, inputs)
                test_loss += loss.item()

                for j in range(inputs.size(0)):
                    # Save images
                    save_image(inputs[j], os.path.join(input_dir, f'input_{idx}.png'))
                    save_image(outputs[j], os.path.join(output_dir, f'output_{idx}.png'))
                    
                    ori_image = to_pil(inputs[j])
                    rm_image = to_pil(outputs[j])
                    
                    psnr_value = calculate_psnr(inputs[j], outputs[j])
                    ssim_value = calculate_ssim(ori_image, rm_image)
                    lpips_value = calculate_lpips_similarity(ori_image, rm_image)
                    
                    psnr += psnr_value
                    ssim += ssim_value
                    lpips += lpips_value
                    
                    # Write metrics for this image to the CSV file
                    csv_writer.writerow([f'input_{idx}', psnr_value, lpips_value, ssim_value])
                    
                    idx += 1

    # Compute averages
    test_loss /= len(dataloader)
    psnr /= idx
    lpips /= idx
    ssim /= idx

    print(f"Test Loss: {test_loss:.4f}, PSNR: {psnr:.4f}, LPIPS: {lpips:.4f}, SSIM: {ssim:.4f}")

    if not is_psnr:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('PSNR', psnr, epoch)
        writer.add_scalar('LPIPS', lpips, epoch)
        writer.add_scalar('SSIM', ssim, epoch)


# In case load model
if args.load_model is not None:
    model.load_state_dict(torch.load(model_save_path))

for epoch in range(num_epochs):
    train(model, train_loader)
    if epoch % 10 == 0:
        test(model, test_loader, epoch)
        torch.save(model.state_dict(), model_save_path)


# PSNR Test for SNR 1-30
snr_values = list(range(1, 31))
for i in snr_values:
    test(model, test_loader, epoch=f"snr_{i}", test_snr=i, is_psnr=True)