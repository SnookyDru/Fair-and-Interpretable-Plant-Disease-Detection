import torch
import torch.nn as nn
import torch.nn.functional as F
from PromptSSM import PromptSSM 

class Downsample2d(nn.Module):
    """Mds: simple stride-2 3x3 conv downsample (keeps channels)."""
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample2d(nn.Module):
    """Mus: bilinear upsample + 3x3 conv (keeps channels)."""
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, padding=1)
    def forward(self, x, size=None):
        if size is None:
            H, W = x.shape[-2]*2, x.shape[-1]*2
        else:
            H, W = size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return self.conv(x)

class DyProMamba(nn.Module):
    """
    Implements Eq. (8):
      Y4 = M3x3( concat( M3x3(X), Y3 ) )
      Y5 = Mus( Mps( Mds( Mps(Y4) ) ) )
      Yr = M3x3( Mps(Y5) )
    """
    def __init__(self, C_hsi=31, in_ch_x=3, H=512, W=512,
                 patch_size=16, embed_dim=256, vim_E=384, vim_N=16, vim_ks=3):
        super().__init__()
        self.C_hsi, self.H, self.W = C_hsi, H, W

        # Map X to latent; fuse with Y3 → Y4
        self.map_x = nn.Conv2d(in_ch_x, C_hsi, kernel_size=3, padding=1)       # M3x3(X) but to C_hsi
        self.fuse_y4 = nn.Conv2d(2 * C_hsi, C_hsi, kernel_size=3, padding=1)   # M3x3(Mcct(...))

        # PromptSSMs (3 stages)
        self.psm1 = PromptSSM(C_hsi, H, W, patch_size, embed_dim, vim_E, vim_N, vim_ks)
        self.down = Downsample2d(C_hsi)                                        # Mds
        self.psm2 = PromptSSM(C_hsi, H//2, W//2, patch_size, embed_dim, vim_E, vim_N, vim_ks)
        self.up   = Upsample2d(C_hsi)                                          # Mus
        self.psm3 = PromptSSM(C_hsi, H, W, patch_size, embed_dim, vim_E, vim_N, vim_ks)

        # Final mapping to reconstructed HSI
        self.final_map = nn.Conv2d(C_hsi, C_hsi, kernel_size=3, padding=1)     # M3x3(Mps(Y5))

    def forward(self, X, Y3, Pspe):
        """
        X   : (B, in_ch_x, H, W)
        Y3  : (B, C_hsi, H, W)
        Pspe: (B, C_hsi, C_hsi)
        Returns:
            Yr, Y4, Y5
        """
        B, _, H, W = X.shape
        assert H == self.H and W == self.W, "DyProMamba expects fixed H,W (for PromptSSM configs)"

        # ---- Y4
        X_lat = self.map_x(X)                         # M3x3(X) → (B,C_hsi,H,W)
        Y4 = self.fuse_y4(torch.cat([X_lat, Y3], 1))  # M3x3(Mcct(...)) → (B,C_hsi,H,W)

        # ---- Y5 = Mus( Mps( Mds( Mps(Y4) ) ) )
        y = self.psm1(Y4, Pspe)                       # Mps(Y4) @ full res
        y_ds = self.down(y)                           # Mds
        y_mid = self.psm2(y_ds, Pspe)                 # Mps at low res
        y_us = self.up(y_mid, size=(H, W))            # Mus back to (H,W)
        Y5 = self.psm3(y_us, Pspe)                    # third Mps at full res

        # ---- Yr = M3x3(Mps(Y5))
        Yr = self.final_map(self.psm3(Y5, Pspe))
        return Yr, Y4, Y5