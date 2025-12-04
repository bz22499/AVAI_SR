# Code inspired by Lab7_intro_to_INR colab notebook, https://github.com/UoB-CS-AVAI
# uses the exact same network setup 


# %pip install torch torchvision pillow scikit-image lpips matplotlib 

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
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# upsampling factor. Set to 0.5 for x16 (relative to x8), 1.0 for x8.
LR_RESIZE_FACTOR = 1.0

# noise level for AWGN added to LR image. Set to 0, 25, or 50
NOISE_SIGMA = 50

# limit number of images to process (set to None for all)
MAX_IMAGES = None


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


class SineLayer(nn.Module):
    """ Linear layer followed by the sine activation

    If `is_first == True`, then it represents the first layer of the network.
    In this case, omega_0 is a frequency factor, which simply multiplies the activations before the nonlinearity.
    Note that it influences the initialization scheme.

    If `is_first == False`, then the weights will be divided by omega_0 so as to keep the magnitude of activations constant,
    but boost gradients to the weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    # initialize weights uniformly
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # 1. pass input through linear layer (self.linear layer performs the linear transformation on the input)
        x = self.linear(input)

        # 2. scale the output of the linear transformation by the frequency factor
        x = x * self.omega_0

        # 3. apply sine activation
        x = torch.sin(x)

        return x

class Siren(nn.Module):
    """ SIREN architecture """

    def __init__(self, in_features, out_features, hidden_features=256, hidden_layers=3, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net) # sequential wrapper of SineLayer and Linear

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

def get_mgrid(sidelen1,sidelen2, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''

    if sidelen1 >= sidelen2:
      # use sidelen1 steps to generate the grid
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen1)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)
      # crop it along one axis to fit sidelen2
      minor = int((sidelen1 - sidelen2)/2)
      mgrid = mgrid[:,minor:sidelen2 + minor]

    if sidelen1 < sidelen2:
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen2)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)

      minor = int((sidelen2 - sidelen1)/2)
      mgrid = mgrid[minor:sidelen1 + minor,:]

    # flatten the gird
    mgrid = mgrid.reshape(-1, dim)

    return mgrid

def image_to_tensor(img):
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


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
    img_LR = img_LR_tensor.to(device)
    img_HR = img_HR_tensor.to(device)
    
    # Apply Resize Factor ( x8, x16)
    if LR_RESIZE_FACTOR != 1.0:
        print(f"Resizing LR Image by factor {LR_RESIZE_FACTOR}")
        img_LR = torch.nn.functional.interpolate(
            img_LR, 
            scale_factor=LR_RESIZE_FACTOR, 
            mode='bicubic', 
            align_corners=True
        )

    # Apply Noise
    if NOISE_SIGMA > 0:
        print(f"Adding Noise (Sigma={NOISE_SIGMA})")
        sigma = NOISE_SIGMA / 255.0
        noise = torch.randn_like(img_LR) * sigma
        img_LR = img_LR + noise
        img_LR = torch.clamp(img_LR, 0, 1)

    # 1 get dimensions
    _, c, lr_h, lr_w = img_LR.shape # lr image is [1, 3, H, W]
    _, _, hr_h, hr_w = img_HR.shape
    
    print(f"HR Image Shape: {img_HR.shape}")
    print(f"LR Input Shape: {img_LR.shape}")

    # 2 create coordinate grids (to use as the inputs)
    lr_coords = get_mgrid(lr_h, lr_w).to(device) # Inputs for training
    hr_coords = get_mgrid(hr_h, hr_w).to(device) # Inputs for super-resolution (inference)

    # 3 format pixel values (the targets for the network)

    # flatten image pixels to match list of coordinates.
    # permute to (H, W, C), then flatten to (N, 3) - i.e. a list of RGB values for each pixel coordinate
    lr_pixels = img_LR.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    hr_pixels = img_HR.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

    # normalise from [0, 1] to [-1, 1]
    lr_pixels = (lr_pixels - 0.5) / 0.5
    hr_pixels = (hr_pixels - 0.5) / 0.5

    print(f"Training Input (Coords): {lr_coords.shape}") # Should be (N_lr, 2)
    print(f"Training Target (Pixels): {lr_pixels.shape}") # Should be (N_lr, 3)

    # initialise model
    model = Siren(in_features=2, out_features=3, hidden_features=256, 
                  hidden_layers=3, outermost_linear=True).to(device)

    # adam as optimiser
    optim_inr = torch.optim.Adam(lr=1e-4, params=model.parameters())

    # training loop
    steps = 2000 # can be increased
    print("Starting training...")

    loss_history = []

    for step in range(steps):
        # 1 input LR coordinates to model
        model_output, _ = model(lr_coords)
        
        # 2 compare output to LR Pixels
        loss = ((model_output - lr_pixels)**2).mean() # MSE Loss

        # 3 optimise
        optim_inr.zero_grad()
        loss.backward()
        optim_inr.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    print("Training Complete.")

    print("Upscaling")

    # query with HR coordinates
    # with torch.no_grad():
    #     sr_output, _ = model(hr_coords)

    # had memory issues with above for large images, so do in chunks
    model.eval()
    chunk_size = 50000
    predictions = []
    
    hr_coords = hr_coords.to(device) 
    num_pixels = hr_coords.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_pixels, chunk_size):
            batch_coords = hr_coords[i : i + chunk_size]
            
            batch_out, _ = model(batch_coords)
            
            predictions.append(batch_out)
    
    sr_output = torch.cat(predictions, dim=0)

    # reshape back to image format
    sr_img = sr_output.view(hr_h, hr_w, 3).cpu().numpy()

    # un-normalise: [-1, 1] back to [0, 1]
    sr_img = (sr_img + 1) / 2
    sr_img = np.clip(sr_img, 0, 1)

    # get ground truth
    gt_img = img_HR.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # PSNR
    mse_val = np.mean((gt_img - sr_img) ** 2)
    sr_psnr = 20 * np.log10(1.0 / np.sqrt(mse_val))
    print(f"Super Resolution PSNR: {sr_psnr:.2f} dB")
    
    dataset_psnr.append(sr_psnr)
    
    # SSIM
    current_ssim = ssim(
        gt_img, 
        sr_img, 
        data_range=1.0, 
        channel_axis=2
    )
    dataset_ssim.append(current_ssim)
    
    # LPIPS needs Tensor inputs (1, 3, H, W) normalised to [-1, 1]
    # sr_output is (N, 3) in range [-1, 1], where n is the number of pixels
    # Reshape sr_output to (1, 3, H, W)
    with torch.no_grad():
        sr_tensor = sr_output.view(1, hr_h, hr_w, 3).permute(0, 3, 1, 2)
        # Ensure it is in [-1, 1] range properly
        sr_tensor = torch.clamp(sr_tensor, -1, 1)
        
        gt_tensor = img_HR.clone() 
        # img_HR is [0, 1], convert to [-1, 1]
        gt_tensor = gt_tensor * 2 - 1
        
        current_lpips = loss_fn_lpips(sr_tensor, gt_tensor).item()
        dataset_lpips.append(current_lpips)
        
    print(f"Image Results: PSNR: {sr_psnr:.2f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}")

# Print averaged results

print(f"Average PSNR:  {np.mean(dataset_psnr):.2f} dB")
print(f"Average SSIM:  {np.mean(dataset_ssim):.4f}")
print(f"Average LPIPS: {np.mean(dataset_lpips):.4f}")