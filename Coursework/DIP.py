# Code inspired by https://github.com/UoB-CS-AVAI/Week2-train-Deep-Neural-Network-to-denoise-image and https://github.com/DmitryUlyanov/deep-image-prior
# 
# The model architectures (skip, unet, resnet) and utils functions are directly copied from these githubs

# %pip install torch torchvision pillow scikit-image lpips matplotlib 


# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from models import *
from utils import *
from utils.sr_utils import *
from utils.common_utils import *

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")



# upsampling factor. Set to 1.0 for x8, 0.5 for x16
LR_RESIZE_FACTOR = 0.5

# noise level for AWGN added to LR image. Set to 0, 25, or 50
NOISE_SIGMA = 0

# limit number of images to process (set to None for all)
MAX_IMAGES = 3


# Get LR dataset and HR dataset (for ground truths)

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



dataset_psnr = []
dataset_ssim = []
dataset_lpips = []

loss_fn_lpips = lpips.LPIPS(net='alex').to(device)




val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print("Starting Validation Loop")

for img_idx, (img_LR_tensor, img_HR_tensor) in enumerate(val_loader):
    
    if MAX_IMAGES is not None and img_idx >= MAX_IMAGES:
        print(f"Reached MAX_IMAGES limit ({MAX_IMAGES}). Stopping.")
        break

    print(f"\nProcessing Image {img_idx+1}/{len(val_dataset)}")
    
    # convert from [C, H,W] to [1, C, H, W] and move to GPU
    img_LR_var = img_LR_tensor.to(device)
    img_HR_var = img_HR_tensor.to(device)

    print(f"HR Image Shape: {img_HR_var.shape}")
    print(f"LR Input Shape: {img_LR_var.shape}")


    # resize the LR image ( x8, x16)
    if LR_RESIZE_FACTOR != 1.0:
        print(f"Resizing LR Image by factor {LR_RESIZE_FACTOR}")
        img_LR_var = torch.nn.functional.interpolate(
            img_LR_var, 
            scale_factor=LR_RESIZE_FACTOR, 
            mode='bicubic', 
            align_corners=True
        )

    TOTAL_SCALE_FACTOR = 8 / LR_RESIZE_FACTOR
    print(f"Effective Super-Resolution: x{TOTAL_SCALE_FACTOR}")

    if NOISE_SIGMA > 0:
        print(f"Adding Noise (Sigma={NOISE_SIGMA})")
        sigma = NOISE_SIGMA / 255.0
        noise = torch.randn_like(img_LR_var) * sigma
        img_LR_var = img_LR_var + noise
        img_LR_var = torch.clamp(img_LR_var, 0, 1)

    print(f"Final Input LR Shape: {img_LR_var.shape}")


    # Define network hyperparameters

    INPUT = 'noise' # choice of 'noise' or 'meshgrid' - just use noise for DIP
    pad = 'reflection' # choice of padding type, which 
    OPT_OVER = 'net' # 'net' - optimise the network weights. 'net,input' - optimise the noise too. 

    reg_noise_std = 1./30. # std of noise added to input at each iteration
    LR = 0.01 # learning rate for optimizer
    OPTIMIZER = 'adam' # 'adam' or 'LBFGS'
    show_every = 100 # how often to show results
    num_iter = 500 # total iterations 
    input_depth = 32 # number of channels in input noise 
    figsize = 4 # figure size for plotting

    NET_TYPE = 'skip' # choice of 'skip', 'resnet' or 'unet'


    # Define input noise, z - dimensions of HR image
    
    # Generate fresh noise for each image in the loop
    net_input = get_noise(input_depth, INPUT, (img_HR_var.shape[2], img_HR_var.shape[3])).type(dtype).detach()

    print("Input noise shape:", net_input.shape)


    # Define network architecture (skip, resnet or unet) and loss function

    def get_net(input_depth, NET_TYPE, pad, skip_n33d=32, skip_n33u=32, skip_n11=4, num_scales=5, upsample_mode='bilinear'):
        if NET_TYPE == 'skip':
            return skip(input_depth, 3, 
                   num_channels_down = [skip_n33d] * num_scales, 
                   num_channels_up =   [skip_n33u] * num_scales,
                   num_channels_skip =    [skip_n11] * num_scales, 
                   filter_size_up = 3, filter_size_down = 3, 
                   upsample_mode=upsample_mode, filter_skip_size=1,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        elif NET_TYPE == 'resnet':
            return resnet.ResNet(input_depth, 3, num_channels=32, num_blocks=8, act_fun='LeakyReLU')
        elif NET_TYPE == 'unet':
            return unet.UNet(num_input_channels=input_depth, num_output_channels=3,
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
        else:
            assert False

    # initialise network
    net = get_net(input_depth, NET_TYPE, pad,
                  skip_n33d=32,
                  skip_n33u=32,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    # print number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # loss function
    mse = torch.nn.MSELoss().type(dtype)


    # Define degradation function, H - known to be bicubic x8 downsampling

    def degradation_operator(hr_tensor): 
        return torch.nn.functional.interpolate(
            hr_tensor, # the tensor of the HR x_hat output by our model
            scale_factor=1/TOTAL_SCALE_FACTOR, 
            mode='bicubic', 
            align_corners=True
        )
    


    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    i = 0

    # define closure function (runs once per iteration inside "optimize")
    def closure():
        global i # Use global instead of nonlocal for script-level variables
        
        # add noise to input for regularisation
        if reg_noise_std > 0:
            # Note: We use net_input_saved from the outer scope
            net_input_active = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input_active = net_input_saved
        
        # forward pass to generate HR estimate
        out_HR = net(net_input_active)
        
        # downsample the hr output to make it the same dimensions as our lr input
        out_LR = degradation_operator(out_HR)
        
        # compare downsampled model output vs original lr input
        total_loss = mse(out_LR, img_LR_var)
        
        # backpropagate loss
        total_loss.backward()

        if i % show_every == 0:
             # Just print loss, don't accumulate history lists
             print(f'Iter {i:05d} | Loss {total_loss.item():.6f}')

        i += 1
        
        return total_loss

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    # get final result
    out_HR_final = np.clip(torch_to_np(net(net_input)), 0, 1)

    img_HR_np = torch_to_np(img_HR_var)

    # PSNR
    mse_val = np.mean((img_HR_np - out_HR_final) ** 2)
    final_psnr = 10 * np.log10(1 / mse_val)
    
    # SSIM
    final_ssim = ssim(
        img_HR_np.transpose(1, 2, 0), 
        out_HR_final.transpose(1, 2, 0), 
        data_range=1.0, 
        channel_axis=2
    )

    # LPIPS
    with torch.no_grad():
        final_lpips = loss_fn_lpips(
            net(net_input) * 2 - 1, 
            img_HR_var * 2 - 1
        ).item()
    
    print(f"Result -> PSNR: {final_psnr:.2f} | SSIM: {final_ssim:.4f} | LPIPS: {final_lpips:.4f}")

    # Store final metrics for this image into global dataset lists
    dataset_psnr.append(final_psnr)
    dataset_ssim.append(final_ssim)
    dataset_lpips.append(final_lpips)


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

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

#display hr
ax[0].imshow(torch_to_np(img_HR_var).transpose(1, 2, 0))
ax[0].set_title(f"Ground Truth HR (Last Image)")
ax[0].axis('off')

# lr
display_LR = torch.nn.functional.interpolate(img_LR_var, size=img_HR_var.shape[2:], mode='nearest')
ax[1].imshow(torch_to_np(display_LR).transpose(1, 2, 0))
ax[1].set_title(f"Input LR")
ax[1].axis('off')

# DIP output
ax[2].imshow(out_HR_final.transpose(1, 2, 0))
ax[2].set_title(f"DIP Output HR\nPSNR: {dataset_psnr[-1]:.2f} dB")
ax[2].axis('off')

plt.show()
