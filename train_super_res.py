import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import argparse, os
from SemanticTestbed.allmodels_superres_nonfeatup import ClassicalInceptionV3, QuantumInceptionV3,  ClassicalResNet, QuantumResNet50, ViT32, QViT32, SwinT, QSwinT
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from semantic_communication.metrics import calculate_psnr, calculate_ssim, calculate_lpips_similarity_tensor, calculate_lpips_similarity
from torch.utils.tensorboard import SummaryWriter

# Pass through params
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='[inceptionv3, qinceptionv3, resnet, \
        qresnet, vit, qvit, swin, qswin]')
    parser.add_argument('--dataset', type=str, default='aider', help='[aider, euro, hrsc2016]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--noise_type', type=str, default='awgn', help='[awgn, rayleigh]')
    return parser.parse_args()

# Parse arguments
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_folder = "dataset/AIDBig"
transform = transforms.Compose([
    transforms.Resize((299, 299)) if args.model_name in ['inceptionv3', 'qinceptionv3'] else transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
batch_size = args.batch_size
num_epochs = 100
train_snr = 15
noise_type = args.noise_type

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
logdir = f"logs/log-{args.model_name}-{timestamp}/" if args.load_model is None else args.load_model
model_save_path = f'{logdir}/final_model.pth'


dataset = datasets.ImageFolder(root=root_folder, transform=transform)
subset_indices = torch.randperm(len(dataset)).tolist()
subset = Subset(dataset, subset_indices)
train_size = int(0.8 * len(subset))
train_dataset, test_dataset = torch.utils.data.random_split(subset, [train_size, len(subset) - train_size])

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(dataset.classes)

# Model selection
if args.model_name == 'inceptionv3':
    model = ClassicalInceptionV3(num_classes).to(device)
elif args.model_name == 'qinceptionv3':
    model = QuantumInceptionV3(num_classes).to(device)
elif args.model_name == 'resnet':
    model = ClassicalResNet(num_classes).to(device)
elif args.model_name == 'qresnet':
    model = QuantumResNet50(num_classes).to(device)
elif args.model_name == 'vit':
    model = ViT32(num_classes).to(device)
elif args.model_name == 'qvit':
    model = QViT32(num_classes).to(device)
elif args.model_name == 'swin':
    model = SwinT(num_classes).to(device)
elif args.model_name == 'qswin':
    model = QSwinT(num_classes).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=logdir)
to_pil = transforms.ToPILImage()

def train(model, dataloader):
    model.train()  # Set the model to training mode
    train_loss = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, train_snr, noise_type)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)
    
    print(f"Train Loss: {train_loss:.4f}")
    writer.add_scalar('Loss/train', train_loss, epoch)


def test(model, dataloader, epoch, test_snr=train_snr, is_psnr=False):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    psnr, lpips, ssim = 0, 0, 0
    
    epoch_dir = os.path.join(logdir, str(epoch))
    
    os.makedirs(epoch_dir, exist_ok=True)
    input_dir = os.path.join(epoch_dir, 'input')
    output_dir = os.path.join(epoch_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    idx = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, test_snr, noise_type)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

            for j in range(inputs.size(0)):
                # Save images
                save_image(inputs[j], os.path.join(input_dir, f'input_{idx}.png'))
                save_image(outputs[j], os.path.join(output_dir, f'output_{idx}.png'))
                
                ori_image = to_pil(inputs[j])
                rm_image = to_pil(outputs[j])
                
                psnr += calculate_psnr(inputs[j], outputs[j])
                ssim += calculate_ssim(ori_image, rm_image)
                lpips += calculate_lpips_similarity(ori_image, rm_image)
                idx += 1

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
    train(model, dataloader_train)
    if epoch % 10 == 0:
        test(model, dataloader_test, epoch)
        torch.save(model.state_dict(), model_save_path)


# PSNR Test for SNR 1-30
snr_values = list(range(1, 31))
for i in snr_values:
    test(model, dataloader_test, epoch=f"snr_{i}", test_snr=i, is_psnr=True)