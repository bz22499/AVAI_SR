# %pip install torch torchvision pillow scikit-image lpips matplotlib 

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import glob
import os
import math
import random

# Standard GPU check
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")



# upsampling factor. Set to 0.5 for x16 (relative to x8), 1.0 for x8.
LR_RESIZE_FACTOR = 0.5

# noise level for AWGN added to LR image. Set to 0, 25, or 50
NOISE_SIGMA = 0

# limit number of images to process (set to None for all)
MAX_IMAGES = 3


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))

        self.transform = transforms.ToTensor() # PIL to Tensor
    
    def __len__(self): # dataloader needs access to length
        return len(self.hr_paths)

    def __getitem__(self, index): # dataloader needs access to dataset items by index
        lr_img = self.transform(Image.open(self.lr_paths[index])) # DIV2K is RGB images
        hr_img = self.transform(Image.open(self.hr_paths[index]))
        return lr_img, hr_img

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'

train_dataset = DIV2KDataset(
    hr_dir=str(DATA_DIR / 'DIV2K_train_HR'),
    lr_dir=str(DATA_DIR / 'DIV2K_train_LR_x8')
)

val_dataset = DIV2KDataset(
    hr_dir=str(DATA_DIR / 'DIV2K_valid_HR'),
    lr_dir=str(DATA_DIR / 'DIV2K_valid_LR_x8')
)


EPOCHS = 15
CROPS_PER_EPOCH = 500   # Number of training examples extracted per epoch
LEARNING_RATE = 0.0005


def degradation(img_tensor, scale=0.5):
    # downsamples an image by "scale" to create a lower-resolution version
    # so the network can learn how to reverse the degradation
    return TF.interpolate(
        img_tensor,
        scale_factor=scale,
        mode='bicubic',
        align_corners=False
    )

class ZSSRInternalDataset(Dataset):
    def __init__(self, target_img, num_samples=1000, crop_size=64):
        self.target = target_img
        self.num_samples = num_samples
        self.crop_size = crop_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # randomly crop the target (our original lr image)
        _, _, h, w = self.target.shape
        
        cs = self.crop_size
        top = random.randint(0, max(0, h - cs - 1))
        left = random.randint(0, max(0, w - cs - 1))

        hr_crop = self.target[:, :, top:top+cs, left:left+cs]

        # create the input by degrading/downsampling the cropped area
        lr_crop = degradation(hr_crop, scale=0.5)

        # squeeze to (C, H, W) for augmentation
        hr_crop = hr_crop.squeeze(0)
        lr_crop = lr_crop.squeeze(0)

        # data augmentation
        if random.random() > 0.5: # Horizontal Flip
            hr_crop = TF.hflip(hr_crop)
            lr_crop = TF.hflip(lr_crop)
        if random.random() > 0.5: # Vertical Flip
            hr_crop = TF.vflip(hr_crop)
            lr_crop = TF.vflip(lr_crop)
        if random.random() > 0.5: # 90-degree Rotation
            hr_crop = torch.rot90(hr_crop, 1, [1, 2])
            lr_crop = torch.rot90(lr_crop, 1, [1, 2])

        return lr_crop, hr_crop

class ZSSRNet(nn.Module):
    def __init__(self, channels=64):
        super(ZSSRNet, self).__init__()
        
        # head
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        
        # body "We use a simple, fully convolutional network, with 8 hidden layers"
        body_layers = []
        for _ in range(8):
            body_layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            body_layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body_layers)
        
        # tail (predicts residual, i.e. corrections)
        self.tail = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # bicubic upsample - "base guess" is direct upsampling - this will be blurry but we can learn improvements
        x_upscaled = TF.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        
        # predict residual (corrections to the blurry upsampled image)
        feat = self.head(x_upscaled)
        feat = self.body(feat)
        residual = self.tail(feat)
        
        # add residual to base 
        return x_upscaled + residual


dataset_psnr = []
dataset_ssim = []
dataset_lpips = []

loss_fn_lpips = lpips.LPIPS(net='alex').to(device)



val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print("Starting Validation Loop")

for img_idx, (img_LR_tensor, img_HR_tensor) in enumerate(val_loader):

    if MAX_IMAGES is not None and img_idx >= MAX_IMAGES:
        print(f"reached MAX_IMAGES limit ({MAX_IMAGES})")
        break

    print(f"\nProcessing Image {img_idx+1}/{len(val_dataset)}")

    # convert from [C, H,W] to [1, C, H, W] and move to GPU
    img_LR_var = img_LR_tensor.to(device)
    img_HR_var = img_HR_tensor.to(device)

    if LR_RESIZE_FACTOR != 1.0:
        print(f"Resizing LR Image by factor {LR_RESIZE_FACTOR}")
        img_LR_var = TF.interpolate(
            img_LR_var, 
            scale_factor=LR_RESIZE_FACTOR, 
            mode='bicubic', 
            align_corners=False
        )

    TOTAL_SCALE_FACTOR = 8 / LR_RESIZE_FACTOR

    # calculate number of x2 passes needed
    NUM_PASSES = int(math.log2(TOTAL_SCALE_FACTOR))
    
    print(f"Effective Super-Resolution: x{int(TOTAL_SCALE_FACTOR)} ({NUM_PASSES} passes of x2)")

    # Apply Noise
    if NOISE_SIGMA > 0:
        print(f"Adding Noise (Sigma={NOISE_SIGMA})")
        sigma = NOISE_SIGMA / 255.0
        noise = torch.randn_like(img_LR_var) * sigma
        img_LR_var = img_LR_var + noise
        img_LR_var = torch.clamp(img_LR_var, 0, 1)

    print(f"Final Input LR Shape: {img_LR_var.shape}")
    print(f"HR Image Shape: {img_HR_var.shape}")

    # zssr is per image 
    # initialise new model for each image
    model_zssr = ZSSRNet().to(device)
    optimizer_zssr = torch.optim.Adam(model_zssr.parameters(), lr=LEARNING_RATE)
    
    # create internal dataset from the specific LR image
    zssr_ds = ZSSRInternalDataset(img_LR_var, num_samples=CROPS_PER_EPOCH)
    zssr_loader = DataLoader(zssr_ds, batch_size=16, shuffle=True)

    loss_history = []
    
    # Training Loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        model_zssr.train()
        
        for i, (lr_batch, hr_batch) in enumerate(zssr_loader):
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            
            output = model_zssr(lr_batch)
            
            # L1 loss like in the original paper
            loss = TF.l1_loss(output, hr_batch)
            
            optimizer_zssr.zero_grad()
            loss.backward()
            optimizer_zssr.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(zssr_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
             print(f"epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
    
    print(f"Running ZSSR Inference x{int(TOTAL_SCALE_FACTOR)}.")
    model_zssr.eval()

    with torch.no_grad():
        current_img = img_LR_var
        
        # 2x upscaling per pass
        for p in range(NUM_PASSES):
            print(f"   Upscaling Pass {p+1}/{NUM_PASSES}...")
            current_img = model_zssr(current_img)
            
        sr_final = current_img
        
        # resize output to match exact HR ground truth dimensions - handles rounding errors
        target_h, target_w = img_HR_var.shape[2], img_HR_var.shape[3]
        
        if sr_final.shape[2:] != (target_h, target_w):
            sr_final = TF.interpolate(
                sr_final, 
                size=(target_h, target_w), 
                mode='bicubic', 
                align_corners=False
            )
        
        sr_final = torch.clamp(sr_final, 0, 1)

    # convert to numpy 
    sr_np = sr_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gt_np = img_HR_var.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # PSNR
    mse_val = np.mean((gt_np - sr_np) ** 2)
    current_psnr = 10 * np.log10(1.0 / mse_val) if mse_val > 0 else 100
    dataset_psnr.append(current_psnr)

    # SSIM
    current_ssim = ssim(gt_np, sr_np, data_range=1.0, channel_axis=2)
    dataset_ssim.append(current_ssim)

    # LPIPS expects (N, 3, H, W) in [-1, 1]
    with torch.no_grad():
        sr_tensor_lpips = sr_final * 2 - 1
        gt_tensor_lpips = img_HR_var * 2 - 1
        current_lpips = loss_fn_lpips(sr_tensor_lpips, gt_tensor_lpips).item()
    dataset_lpips.append(current_lpips)

    print(f" image {img_idx+1}  PSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}")


# Print averaged results
print("\n" + "="*40)
print(f"VALIDATION COMPLETE")
print(f"Processed {len(dataset_psnr)} images.")
print("="*40)
print(f"Average PSNR:  {np.mean(dataset_psnr):.2f} dB")
print(f"Average SSIM:  {np.mean(dataset_ssim):.4f}")
print(f"Average LPIPS: {np.mean(dataset_lpips):.4f}")
print("="*40)


# Visualisation (Plots only the last image processed) 

plt.figure(figsize=(18, 6))

lr_np = img_LR_var.squeeze(0).permute(1, 2, 0).cpu().numpy()

plt.subplot(1, 3, 1)
plt.imshow(lr_np)
plt.title(f"LR Input (x{int(TOTAL_SCALE_FACTOR)} smaller)\n{lr_np.shape}")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sr_np)
plt.title(f"ZSSR Output\nPSNR: {dataset_psnr[-1]:.2f} dB")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gt_np)
plt.title(f"Ground Truth\n{gt_np.shape}")
plt.axis('off')

plt.show()