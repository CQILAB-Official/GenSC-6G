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

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='[inceptionv3, resnet, \
        qresnet, vit, swin, mobilenet, efficientnet, vgg, qcnn]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
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
num_epochs = 100

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 512 * 7 * 7),  # Adjusting to match the starting feature map size
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 112x112 -> 224x224
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)  # Reshape to (batch_size, 512, 7, 7)
        x = self.deconv(x)
        return x

class EnhancedDecoder(nn.Module):
    def __init__(self):
        super(EnhancedDecoder, self).__init__()
        
        # Fully connected layer to reshape the latent vector
        self.fc = nn.Sequential(
            nn.Linear(1000, 512 * 7 * 7),
            nn.ReLU()
        )

        # First convolutional block followed by Pixel Shuffle
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)  # 7x7 -> 14x14
        
        # Residual Block 1
        self.res_block1 = self._residual_block(128)

        # Second convolutional block followed by Pixel Shuffle
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)  # 14x14 -> 28x28
        
        # Residual Block 2
        self.res_block2 = self._residual_block(64)

        # Third convolutional block followed by Pixel Shuffle
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pixel_shuffle3 = nn.PixelShuffle(upscale_factor=2)  # 28x28 -> 56x56

        # Fourth convolutional block followed by Pixel Shuffle
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pixel_shuffle4 = nn.PixelShuffle(upscale_factor=2)  # 56x56 -> 112x112

        # Fifth convolutional block followed by Pixel Shuffle (for final upscale to 224x224)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pixel_shuffle5 = nn.PixelShuffle(upscale_factor=2)  # 112x112 -> 224x224

        # Final layer to map to the output image size
        self.final_layer = nn.Conv2d(8, 3, kernel_size=3, padding=1)  # Output RGB image (224x224)

    def forward(self, x):
        # Fully connected layer to reshape the latent vector to feature map
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)

        # Apply the convolutional blocks and pixel shuffle with residual connections
        x = self.pixel_shuffle1(self.conv1(x))
        x = self.res_block1(x) + x  # Residual connection

        x = self.pixel_shuffle2(self.conv2(x))
        x = self.res_block2(x) + x  # Residual connection

        x = self.pixel_shuffle3(self.conv3(x))
        x = self.pixel_shuffle4(self.conv4(x))

        # Additional pixel shuffle block to upscale from 112x112 to 224x224
        x = self.pixel_shuffle5(self.conv5(x))

        # Final convolution to produce the RGB image
        x = self.final_layer(x)

        return x

    def _residual_block(self, channels):
        """Returns a basic residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

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
num_classes = len(torch.unique(train_labels))  # Number of classes

model = EnhancedDecoder().to(device)

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
        outputs = model(features)
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
                outputs = model(features)
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
# snr_values = list(range(1, 31))
# for i in snr_values:
#     test(model, test_loader, epoch=f"snr_{i}", test_snr=i, is_psnr=True)