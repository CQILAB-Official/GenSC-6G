import torch
import numpy as np
import lpips
from skimage.metrics import structural_similarity
import cv2
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import pytorch_fid
import os
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

# Calculate PSNR
def calculate_psnr(original_signal, roi_signal, theta=1.0):
    mse = torch.mean(torch.square(original_signal - roi_signal))
    
    signal_power = torch.square(torch.max(original_signal))
    noise_power = mse * theta # + mse_roni * (1 - theta)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def calculate_compression_rate(original_size, compressed_size):
    return compressed_size / original_size


def calculate_psnr_np(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

# SSIM Metric
def calculate_ssim(image1, image2):
    # Convert images to PIL images
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Compute SSIM between two images
    (score, diff) = structural_similarity(image1, image2, full=True)
    return score

model = lpips.LPIPS(net='alex')
def calculate_lpips_similarity(image1, image2):
    # Convert images to tensors
    tensor1 = image_to_tensor(image1)
    tensor2 = image_to_tensor(image2)
    
    # Calculate LPIPS similarity
    with torch.no_grad():
        similarity = model(tensor1, tensor2).item()

    return similarity

def calculate_lpips_similarity_tensor(tensor1, tensor2):
    with torch.no_grad():
        similarity = model(tensor1, tensor2).item()

    return similarity

def image_to_tensor(image):
    img = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = img * 2 - 1
    return img

def calculate_fid_score(path1, path2, batch_size, device=torch.device("cuda:0")):
    fid = pytorch_fid.fid_score.calculate_fid_given_paths([path1, path2], dims=2048, batch_size=batch_size, device=device)
    return fid

def calculate_inception_score(path1, device="cuda:0"):
    inception = InceptionScore()
    images = load_images(path1)
    num_images, image_channels, image_height, image_width = images.size(0), images.size(1), images.size(2), images.size(3)
    images = images.view(num_images, image_channels, image_height, image_width)
    images = images.to(torch.uint8)
    inception.update(images)
    return inception.compute()


def calculate_miou(gt_image, test_image):
    gt_image = np.array(gt_image)
    test_image = np.array(test_image)
    # Flatten the images
    gt_image_flatten = gt_image.flatten()
    test_image_flatten = test_image.flatten()

    # Create a confusion matrix
    cm = confusion_matrix(gt_image_flatten, test_image_flatten)

    # Calculate Intersection over Union (IoU) for each class
    intersection = np.diag(cm)
    ground_truth_union = np.sum(cm, axis=1)
    predicted_union = np.sum(cm, axis=0)
    union = ground_truth_union + predicted_union - intersection

    # Avoid division by zero
    union[union == 0] = 1e-9

    IoU = intersection / union

    # Calculate the mean Intersection over Union (MIoU)
    mIoU = np.mean(IoU)
    return mIoU

def load_images(folder_path):
    target_size = (512, 512)
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            image = image.resize(target_size)  # Resize image to target size
            image = transforms.ToTensor()(image)
            image_list.append(image)
    images = torch.stack(image_list)
    return images