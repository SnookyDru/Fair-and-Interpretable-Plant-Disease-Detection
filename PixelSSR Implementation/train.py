import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PixelSSR import PixelSSR
from dataset import load_rgb_hsi_images, convert_hwb_to_bhw
from dataloader import create_dataloader
from gamma import fit_gamma_distribution, generate_gamma_cube, apply_gamma_mask

# -----------------------------
# Utility: Evaluation Metrics
# -----------------------------
def compute_rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()

def compute_psnr(y_true, y_pred, eps=1e-8):
    mse = F.mse_loss(y_pred, y_true).item()
    return -20.0 * np.log10(np.sqrt(mse) + eps)

def compute_ssim(y_true, y_pred):
    # y_true, y_pred: (B, C, H, W)
    y_true = y_true.squeeze().cpu().numpy()
    y_pred = y_pred.detach().squeeze().cpu().numpy()
    bands = y_true.shape[0]
    ssim_vals = []
    for b in range(bands):
        ssim_val = ssim(y_true[b], y_pred[b], data_range=y_true[b].max() - y_true[b].min())
        ssim_vals.append(ssim_val)
    return np.mean(ssim_vals)

def compute_sam(y_true, y_pred, eps=1e-8):
    # spectral angle mapper across channels
    B, C, H, W = y_true.shape
    y_true = y_true.view(B, C, -1)  # (B,C,N)
    y_pred = y_pred.view(B, C, -1)
    dot = torch.sum(y_true * y_pred, dim=1)
    norm_true = torch.norm(y_true, dim=1)
    norm_pred = torch.norm(y_pred, dim=1)
    cos = dot / (norm_true * norm_pred + eps)
    angles = torch.acos(torch.clamp(cos, -1, 1)) * 180.0 / np.pi
    return torch.mean(angles).item()

# -----------------------------
# Loss Functions
# -----------------------------
class HSI_Loss(nn.Module):
    def __init__(self, beta1=1.0, beta2=0.3, beta3=0.3):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.l1 = nn.L1Loss()

        # Pixel2Image mapping module
        self.map_h2r = nn.Sequential(
            nn.Conv2d(31, 31, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(31, 3, kernel_size=3, padding=1)
        )

    def forward(self, X, Yr, Yp, Y3):
        # LpHSI
        LpHSI = self.l1(Yr, Yp)

        # Lrep
        S = nn.Conv2d(31, 3, kernel_size=1).to(X.device)  # simple spectral projection
        Lrep = self.l1(X, S(Yr))

        # LSSIM
        ssim_loss = 1 - compute_ssim(Yp, Yr)

        # LP2I
        LP2I = self.l1(X, self.map_h2r(Y3))

        # Total
        L = LpHSI + self.beta1 * Lrep + self.beta2 * ssim_loss + self.beta3 * LP2I
        return L, {"LpHSI": LpHSI.item(), "Lrep": Lrep.item(), "LSSIM": ssim_loss, "LP2I": LP2I.item()}

# -----------------------------
# Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
rgb_list, hsi_list = load_rgb_hsi_images("dataset")
hsi_list_bhw, rgb_list_bhw = convert_hwb_to_bhw(hsi_list, rgb_list)
dataset, dataloader = create_dataloader(rgb_list_bhw, hsi_list_bhw, batch_size=1, num_workers=0)

# Fit Gamma distribution and generate cube
params, _ = fit_gamma_distribution(hsi_list_bhw, max_samples=5000, normalize=True)
gamma_cube = generate_gamma_cube(512, 512, 31, params, seed=42)
masked_cube, mask = apply_gamma_mask(gamma_cube, ratio=0.0001, seed=42)
masked_cube = masked_cube.unsqueeze(0).to(device)

# Model & Loss
model = PixelSSR(out_bands=31).to(device)
criterion = HSI_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.99), eps=1e-8)

# Scheduler with power decay (1.5)
lambda_lr = lambda epoch: (1.0 / ((epoch + 1) ** 1.5))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 200
print("Starting training...\n")

start_time = time.time()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for rgb_batch, hsi_batch in dataloader:
        rgb_batch, hsi_batch = rgb_batch.to(device), hsi_batch.to(device)

        optimizer.zero_grad()
        Yr, Y3 = model(rgb_batch, masked_cube)   # PixelSSR forward returns (HSI reconstruction, Y3)

        loss, loss_dict = criterion(rgb_batch, Yr, hsi_batch, Y3)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(dataloader)

    # ----------------- Validation (on one batch for speed)
    model.eval()
    with torch.no_grad():
        rgb_val, hsi_val = next(iter(dataloader))
        rgb_val, hsi_val = rgb_val.to(device), hsi_val.to(device)
        Yr_val, Y3_val = model(rgb_val, masked_cube)

        rmse = compute_rmse(hsi_val, Yr_val)
        psnr = compute_psnr(hsi_val, Yr_val)
        ssim_val = compute_ssim(hsi_val, Yr_val)
        sam = compute_sam(hsi_val, Yr_val)

    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.6f} | RMSE: {rmse:.4f} | PSNR: {psnr:.2f} dB | "
          f"SSIM: {ssim_val:.4f} | SAM: {sam:.4f}")

end_time = time.time()
print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes.")
