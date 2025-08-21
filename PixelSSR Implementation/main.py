import os
import time
from dataset import load_rgb_hsi_images, convert_hwb_to_bhw
from dataloader import create_dataloader
from gamma import fit_gamma_distribution, generate_gamma_cube, apply_gamma_mask
import torch
from PixelSSR import PixelSSR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- START TOTAL TIMER --------------------
total_start = time.time()

print(f"Using device: {device}")
print("------------------------------------------------")

print("Loading RGB and HSI images...")
print("------------------------------------------------")
main_dir = "dataset" 
rgb_list, hsi_list = load_rgb_hsi_images(main_dir)

print("Converting RGB and HSI images from HWC to BHW dimension order...")
print("------------------------------------------------")
# Convert HSI and RGB images from HWC to BHW dimension order
hsi_list_bhw, rgb_list_bhw = convert_hwb_to_bhw(hsi_list, rgb_list)

"""
Create DataLoader:
    RGB batch: torch.Size([Batch, 3, H, W])
    HSI batch: torch.Size([Batch, Bands, H, W])
"""
print("Creating DataLoader...")
print("------------------------------------------------")
dataset, dataloader = create_dataloader(rgb_list_bhw, hsi_list_bhw, batch_size=1, num_workers=0)

print("Fitting Gamma distribution to HSI data...")
print("------------------------------------------------")
# Fit Gamma distribution to the HSI data
params, fit_vals = fit_gamma_distribution(hsi_list_bhw, max_samples=5000, normalize=True)

print("Generating synthetic Gamma-modeled hyperspectral cube...")
print("------------------------------------------------")
# Generate a synthetic Gamma-modeled hyperspectral cube
gamma_cube = generate_gamma_cube(512, 512, 31, params, seed=42)

print("Applying mask to the Gamma-modeled hyperspectral cube...")
print("------------------------------------------------")
# Apply a mask to the Gamma-modeled cube
masked_cube, mask = apply_gamma_mask(gamma_cube, ratio=0.0001, seed=42)  # 0.0001%
masked_cube = masked_cube.unsqueeze(0)  # Add batch dimension
# masked_cube shape: (1, 31, 512, 512)

for rgb_batch, hsi_batch in dataloader:
    print(f"RGB batch shape: {rgb_batch.shape}")  # Check RGB shape
    print(f"HSI batch shape: {hsi_batch.shape}")  # Check HSI shape

    # -------------------- PIXELSSR TIMER --------------------
    pixelssr_start = time.time()
    print("Creating PixelSSR model...")
    print("------------------------------------------------")
    PixelSSR_model = PixelSSR(out_bands=31).to(device)

    print("Running PixelSSR model...")
    print("------------------------------------------------")
    
    Yr, Y3 = PixelSSR_model(rgb_batch.to(device), masked_cube.to(device))
    pixelssr_end = time.time()
    pixelssr_time = pixelssr_end - pixelssr_start
    print(f"PixelSSR forward pass took: {pixelssr_time:.2f} seconds")
    break

print("RGB to HSI conversion via PixelSSR completed")
print("------------------------------------------------")
print("Output shape of Yr (HSI):", Yr.shape)  # (1, 31, 512, 512)

# -------------------- END TOTAL TIMER --------------------
total_end = time.time()
total_time = total_end - total_start
print(f"Total runtime of the whole script: {total_time:.2f} seconds")

# -------------------- SAVE EACH BAND AS IMAGE --------------------
output_dir = "output_hsi_bands"
os.makedirs(output_dir, exist_ok=True)

Yr_cpu = Yr.squeeze(0).detach().cpu()  # (31, 512, 512)
print(f"Saving {Yr_cpu.shape[0]} spectral bands to '{output_dir}'")

for i in range(Yr_cpu.shape[0]):
    band_img = Yr_cpu[i, :, :].numpy()
    plt.imsave(os.path.join(output_dir, f"band_{i+1:02d}.png"), band_img, cmap="gray")

print("Saved all bands as grayscale PNGs.")
