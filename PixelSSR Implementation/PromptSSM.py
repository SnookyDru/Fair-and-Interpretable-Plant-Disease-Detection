import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpaFRFT import MScanBlock  

class PromptSSM(nn.Module):
    """
    One PromptSSM:
      - Spectral-prompt SSA (Eq. 9): F3 = V · (Pspe · softmax(K^T Q))
      - Spatial SSM (ViM) on the resulting map
      - FFN + residual
    """
    def __init__(self, C, H, W, patch_size=16, embed_dim=256, vim_E=384, vim_N=16, vim_ks=3):
        super().__init__()
        self.C, self.H, self.W = C, H, W

        # Q,K,V via 3x3 convs (stay in (B,C,H,W)), then we flatten to (B,HW,C)
        self.q_conv = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.k_conv = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.v_conv = nn.Conv2d(C, C, kernel_size=3, padding=1)

        # Spatial SSM (Vision Mamba) over the feature map
        self.mscan = MScanBlock(in_ch=C, patch_size=patch_size, embed_dim=embed_dim,
                                H=H, W=W, vim_E=vim_E, vim_N=vim_N, vim_ks=vim_ks)

        # FFN head
        self.ffn = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
        )

    def forward(self, F_in, Pspe):
        """
        F_in : (B,C,H,W)
        Pspe : (B,C,C)  — spectral prompt matrix (from SpectralWiseMSA)
        """
        B, C, H, W = F_in.shape
        assert C == self.C and H == self.H and W == self.W, "PromptSSM shape mismatch"

        # --- Q,K,V from 3x3
        Qm = self.q_conv(F_in)                     # (B,C,H,W)
        Km = self.k_conv(F_in)                     # (B,C,H,W)
        Vm = self.v_conv(F_in)                     # (B,C,H,W)

        # Flatten spatial → (B,HW,C)
        HW = H * W
        Q = Qm.permute(0, 2, 3, 1).reshape(B, HW, C)   # (B,HW,C)
        K = Km.permute(0, 2, 3, 1).reshape(B, HW, C)   # (B,HW,C)
        V = Vm.permute(0, 2, 3, 1).reshape(B, HW, C)   # (B,HW,C)

        # --- Eq. (9): F3 = V · (Pspe · softmax(K^T Q))
        # K^T Q over HW → (B,C,C)
        attn_logits = torch.matmul(K.transpose(1, 2), Q) / math.sqrt(HW)  # (B,C,C)
        A = F.softmax(attn_logits, dim=-1)                                 # Msft(K^T Q)
        M = torch.matmul(Pspe, A)                                          # Pspe · softmax(...)
        F3 = torch.matmul(V, M)                                            # (B,HW,C)

        # Back to (B,C,H,W)
        F3_map = F3.view(B, H, W, C).permute(0, 3, 1, 2)

        # --- Spatial SSM (Vision Mamba)
        F_ssm = self.mscan(F3_map)                                         # (B,C,H,W)

        # --- FFN + residual
        out = F_ssm + self.ffn(F_ssm)
        return out
