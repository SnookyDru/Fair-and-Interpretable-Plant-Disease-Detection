import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralWiseMSA(nn.Module):
    """
    Spectral-wise Multi-head Self-Attention (S-MSA)
    Inputs:
        X_in : (B, C, H, W)   # feature map sorted by wavelength along C
    Outputs:
        X_out: (B, C, H, W)   # refined features
        Pspe : (B, C, C)      # spectral prompt matrix per sample (block-diag of per-head attentions)

    Notes:
    - Attention is along the spectral (channel) dimension.
    - Q,K,V are (B, HW, C); each head is (B, HW, d_h).
    - A_j = softmax(σ_j * K_j^T @ Q_j) ∈ R^{d_h × d_h}
    - head_j = V_j @ A_j ∈ R^{HW × d_h}; concat over heads → (B, HW, C)
    """
    def __init__(self, C: int, heads: int = 4):
        super().__init__()
        assert C % heads == 0, "C must be divisible by number of heads"
        self.C = C
        self.heads = heads
        self.dh = C // heads

        # Linear projections over spectral channels (last dim)
        self.w_q = nn.Linear(C, C, bias=False)
        self.w_k = nn.Linear(C, C, bias=False)
        self.w_v = nn.Linear(C, C, bias=False)

        # Per-head learnable scaling σ_j
        self.sigmas = nn.Parameter(torch.ones(heads))

        # Final projection W ∈ R^{C×C}
        self.w_out = nn.Linear(C, C, bias=False)

        # Positional embedding fp(·): two depth-wise conv3×3 + GELU
        # Applied to V reshaped to (B,C,H,W), then reshaped back to (B,HW,C)
        self.pos_dw1 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=True)
        self.pos_dw2 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=True)
        self.pos_act = nn.GELU()

    def forward(self, X_in: torch.Tensor):
        B, C, H, W = X_in.shape
        HW = H * W

        # ---- 1) Flatten spatial → tokens across spectral channels
        # (B,C,H,W) -> (B,HW,C)
        X = X_in.permute(0, 2, 3, 1).reshape(B, HW, C)

        # ---- 2) Linear projections
        Q = self.w_q(X)   # (B,HW,C)
        K = self.w_k(X)   # (B,HW,C)
        V = self.w_v(X)   # (B,HW,C)

        # ---- 3) Split into heads along spectral channels
        # (B,HW,C) -> list of (B,HW,dh)
        Qh = Q.split(self.dh, dim=-1)
        Kh = K.split(self.dh, dim=-1)
        Vh = V.split(self.dh, dim=-1)

        heads_out = []
        blocks = []  # to assemble Pspe as block-diagonal
        for j in range(self.heads):
            Qj = Qh[j]                      # (B,HW,dh)
            Kj = Kh[j]                      # (B,HW,dh)
            Vj = Vh[j]                      # (B,HW,dh)

            # A_j = softmax(σ_j * K_j^T @ Q_j) over last dim
            # (B,dh,HW) @ (B,HW,dh) -> (B,dh,dh)
            attn_logits = torch.matmul(Kj.transpose(1, 2), Qj)  # (B,dh,dh)
            attn = F.softmax(self.sigmas[j] * attn_logits, dim=-1)  # (B,dh,dh)

            # head_j = V_j @ A_j  -> (B,HW,dh)
            head_j = torch.matmul(Vj, attn)  # (B,HW,dh)
            heads_out.append(head_j)
            blocks.append(attn)               # keep for Pspe

        # ---- 4) Concat heads and final projection
        Hcat = torch.cat(heads_out, dim=-1)         # (B,HW,C)
        Hproj = self.w_out(Hcat)                    # (B,HW,C)

        # ---- 5) Positional embedding fp(V)
        V_spatial = V.view(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)
        pos = self.pos_dw2(self.pos_act(self.pos_dw1(V_spatial)))  # (B,C,H,W)
        pos = pos.permute(0, 2, 3, 1).reshape(B, HW, C)            # (B,HW,C)

        # ---- 6) Output + reshape back
        X_out = Hproj + pos                              # (B,HW,C)
        X_out = X_out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)

        # ---- 7) Spectral prompt P_spe ∈ R^{B×C×C} as block-diagonal of per-head A_j
        # Build a block-diagonal matrix from N blocks (each B×dh×dh)
        Pspe = X_in.new_zeros(B, C, C)
        offset = 0
        for j in range(self.heads):
            dh = self.dh
            Pspe[:, offset:offset+dh, offset:offset+dh] = blocks[j]
            offset += dh

        return X_out, Pspe


class SpectralPrompt(nn.Module):
    """
    Spectral Prompt Pspe construction.
    
    Inputs:
        X     : (B,3,H,W)    # original RGB image
        Ypg   : (B,C,H,W)    # gamma-modeled spectra (point spectra with channels=C)
        Pspa  : (B,3,H,W)    # spatial prompt (RGB-like)
        Phf   : (B,3,H,W)    # high-frequency prompt (RGB-like)
    
    Output:
        Y3    : (B,C,H,W)    # preliminary HSI reconstruction
        Pspe : (B,C,C)      # spectral prompt matrix per sample (block-diag of per-head attentions)
    """
    def __init__(self, C=31, heads = 1):  # C = number of spectral bands (example 31 for HSI)
        super().__init__()
        # For Y1 creation
        self.conv1 = nn.Conv2d(C + 1, C, kernel_size=3, padding=1)  # concat(Ypg, Xm)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        
        # For Y2 and Y3 refinement
        self.conv_spa = nn.Conv2d(C + 3, C, kernel_size=3, padding=1)   # concat Y1, Pspa
        self.conv_hf  = nn.Conv2d(C + 3, C, kernel_size=3, padding=1)   # concat Y2, Phf
        
        self.act = nn.GELU()

        self.ssa = SpectralWiseMSA(C=C, heads=heads)

    def forward(self, X, Ypg, Pspa, Phf):
        # Step 1: Xm = mean RGB
        Xm = torch.mean(X, dim=1, keepdim=True)   # (B,1,H,W)

        # Step 2: Y1
        cat1 = torch.cat([Ypg, Xm], dim=1)        # (B,C+1,H,W)
        Y1 = self.conv1(cat1)
        Y1 = self.act(Y1)
        Y1 = self.conv2(Y1)                       # (B,C,H,W)

        # Step 3: Y2 (with Pspa)
        cat2 = torch.cat([Y1, Pspa], dim=1)       # (B,C+3,H,W)
        Y2 = self.conv_spa(cat2)                  # (B,C,H,W)

        # Step 4: Y3 (with Phf)
        cat3 = torch.cat([Y2, Phf], dim=1)        # (B,C+3,H,W)
        Y3 = self.conv_hf(cat3)                   # (B,C,H,W)

        # After you compute Y3 via your SpectralPrompt pipeline:
        # Y3: (B, C, H, W)  # C = number of HSI bands (e.g., 31)
        Y3_refined, Pspe = self.ssa(Y3)   # Y3_refined: (B,C,H,W), Pspe: (B,C,C)


        return Pspe, Y3, Y1, Y2
