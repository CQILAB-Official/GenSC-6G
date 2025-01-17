from semantic_communication.diffusion_super_res import DiffusionSuperRes, DiffusionUpscaler
import torch
import os
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--device', type=int, default=0, help='[0, 1]')
    parser.add_argument('--snr', type=int, default=30, help='Signal-to-noise ratio for dataset path')
    return parser.parse_args()

args = parse_args()

images_folder_input = f"logs/upsampling/log-vit/snr_{args.snr}/output"
images_folder_output = f"logs/upsampling/log-vit/snr_{args.snr}/diffusion"

# Make output folder if it doesn't exist
os.makedirs(images_folder_output, exist_ok=True)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

diffusion_super_res = DiffusionSuperRes(device)
# diffusion_upscaler = DiffusionUpscaler(device)

for filename in os.listdir(images_folder_input):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(images_folder_input, filename)

        # Open the image using PIL
        img_pil = Image.open(img_path).convert('RGB')

        # Run the diffusion model (using 200 as the inference step count)
        output_img = diffusion_super_res.inference(img_pil, 200)

        # Save the output image in the output folder with the same filename
        output_path = os.path.join(images_folder_output, filename)
        output_img.save(output_path)

        print(f"Processed and saved: {output_path}")