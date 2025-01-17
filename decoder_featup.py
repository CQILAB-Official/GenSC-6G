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
from tqdm import tqdm
import pandas as pd
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.utils import save_image
from datetime import datetime
from PIL import Image

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='resnet50', help='[resnet50, vit, \
        dinov2]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--device', type=int, default=1, help='[0, 1]')
    parser.add_argument('--snr', type=int, default=10, help='Signal-to-noise ratio for dataset path')
    return parser.parse_args()

# Parse arguments
args = parse_args()

model_name = args.model_name
num_epoch = 51

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
logdir = f"logs/upsample-featup/log-{args.model_name}/" if args.load_model is None else args.load_model
model_save_path = f'{logdir}/final_model.pth'

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, model_name='resnet50'):
        super(Decoder, self).__init__()
        
        # Load the FeatUp model for upsampling only
        self.featup = torch.hub.load("mhamilton723/FeatUp", model_name, use_norm=False).to(device)
        self.linear = nn.Linear(1000, 972)
        

    def forward(self, x, guidance):
        # Pass through the upsampler with guidance
        x = self.linear(x)
        x = x.view(-1, 3, 18, 18)
        x1 = self.featup.upsampler.up1(x, guidance)
        x2 = self.featup.upsampler.up2(x1, guidance)
        x3 = self.featup.upsampler.up3(x2, guidance)
        x4 = self.featup.upsampler.up4(x3, guidance)
            
        # Apply convolution layers to transform to RGB image
        # x = self.conv_layers(x)
        return x4

model_parsed = args.model_name

# Load the features, labels, and filenames
train_feature_path = f"GenSC-Testbed/AWGN_Generated_Upsampling/Train/{args.snr}/{model_parsed}_features.npy"
train_label_path = f"GenSC-Testbed/Extracted_Features_Classification/Train/efficientnet_labels.npy"
train_features = np.load(train_feature_path)
train_labels = np.load(train_label_path)
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

test_feature_path = f"GenSC-Testbed/AWGN_Generated_Upsampling/Test/{args.snr}/{model_parsed}_features.npy"
test_label_path = f"GenSC-Testbed/Extracted_Features_Classification/Test/efficientnet_labels.npy"
test_features = np.load(test_feature_path)
test_labels = np.load(test_label_path)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_image_path = f"GenSC-Testbed/AWGN_Generated_Upsampling/Train/{args.snr}/{model_parsed}_filenames.npy"
test_image_path = f"GenSC-Testbed/AWGN_Generated_Upsampling/Test/{args.snr}/{model_parsed}_filenames.npy"
train_image_filenames = np.load(train_image_path)
test_image_filenames = np.load(test_image_path)

print(len(train_features), len(train_labels), len(train_image_filenames))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_paths(image_filenames):
    image_tensors = []
    for img_path in image_filenames:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image)
        image_tensors.append(image_tensor)
    return torch.stack(image_tensors)

# Load images based on file paths
train_image_tensor = load_images_from_paths(train_image_filenames)
test_image_tensor = load_images_from_paths(test_image_filenames)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_features, train_labels, train_image_tensor)
test_dataset = TensorDataset(test_features, test_labels, test_image_tensor)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Set up the device, model, loss function, and optimizer
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

model = Decoder().to(device)

# Print the model's number of parameters before running the test
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=logdir)
to_pil = transforms.ToPILImage()

def train(model, dataloader):
    model.train()  # Set the model to training mode
    train_loss = 0
    for features, labels, inputs in tqdm(dataloader, desc="Training"):
        features, labels, inputs = features.to(device), labels.to(device), inputs.to(device)
        optimizer.zero_grad()
        outputs = model(features, inputs)
        loss = criterion(outputs, inputs)
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
            for features, labels, inputs in tqdm(dataloader, desc="Evaluating"):
                features, labels, inputs = features.to(device), labels.to(device), inputs.to(device)
                outputs = model(features, inputs)
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

for epoch in range(num_epoch):
    train(model, train_loader)
    if epoch % 10 == 0:
        test(model, test_loader, epoch)

