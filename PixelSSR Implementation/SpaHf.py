import torch
import torch.nn as nn
import torch.nn.functional as F

class SpaHF(nn.Module):
    """
    Spa-HF Prompt (P_hf)
    F2 = up(avgpool(F1))
    P_hf = Conv3x3(GELU(Conv3x3(F1 - F2)))

    Args:
        C:        # channels of F1
        pool:     pooling factor (int or tuple). e.g., 2, 4
        up_mode:  'bilinear' (default) or 'nearest'
    """
    def __init__(self, C, pool=2, up_mode='bilinear', H=512, W=512):
        super().__init__()
        self.H = H
        self.W = W

        self.avg = nn.AvgPool2d(kernel_size=pool, stride=pool)
        # upsample back to original size
        if up_mode == 'bilinear':
            self.up = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        else:
            self.up = lambda x, size: F.interpolate(x, size=size, mode='nearest')

        self.refine = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.GELU(),                      # M_GeLU
            nn.Conv2d(C, 3, kernel_size=3, padding=1)
        )

    def forward(self, F1):                   # F1: (B, C, H, W)
        F2_low = self.avg(F1)                # low-pass
        F2 = self.up(F2_low, size=(self.H, self.W))    # back to (H,W)
        hf = F1 - F2                         # high-frequency residue
        Phf = self.refine(hf)                # two 3x3 + GeLU
        return Phf, F2, hf                   # return extras if you want to visualize
