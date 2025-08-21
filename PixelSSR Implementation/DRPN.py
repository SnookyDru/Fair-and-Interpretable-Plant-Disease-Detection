import torch
import torch.nn as nn
from SpaFRFT import SpaFRFT
from SpaHf import SpaHF
from SpectralPrompt import SpectralPrompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DRPN(torch.nn.Module):
    def __init__(self, in_ch=3, C=40, patch_size=16, embed_dim=256, H=512, W=512,
                 vim_E=384, vim_N=16, vim_ks=3, frft_a0=0.5, frft_a1=0.5, out_bands=31):
        super().__init__()
        self.in_ch = in_ch
        self.C = C
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.H = H
        self.W = W
        self.vim_E = vim_E
        self.vim_N = vim_N
        self.vim_ks = vim_ks
        self.frft_a0 = frft_a0
        self.frft_a1 = frft_a1
        self.up_mode = 'bilinear'  # Default upsampling mode
        self.out_bands = out_bands

        self.F1 = nn.Conv2d(self.in_ch, self.C, kernel_size=3, padding=1).to(device)  # Convolution layer to process input
        self.pspe = SpaFRFT(in_ch=self.C, C=self.C, patch_size=self.patch_size, embed_dim=self.embed_dim, 
                            H=self.H, W=self.W, vim_E=self.vim_E, vim_N=self.vim_N, vim_ks=self.vim_ks, 
                            frft_a0=self.frft_a0, frft_a1=self.frft_a1).to(device)  # SpaFRFT module for spatial and FRFT processing
        
        self.spa_hf = SpaHF(C=self.C, pool=2, up_mode=self.up_mode, H=self.H, W=self.W).to(device)  # SpaHF module for high-frequency processing

        self.spectral_prompt = SpectralPrompt(C=self.out_bands, heads=1).to(device)  # Spectral prompt module for spectral reconstruction

    def forward(self, x, masked_gamma_cube):
        # F1 = M3x3(x)
        F1 = self.F1(x)  # Assuming F1 is a convolution layer

        pspa = self.pspe(F1)  # Apply SpaFRFT
        Phf, F2, hf = self.spa_hf(F1) # Apply SpaHF

        Ypg = masked_gamma_cube  # Use the masked gamma cube as point 
        
        Pspe, Y3, Y1, Y2 = self.spectral_prompt(x, Ypg, pspa, Phf)  # Apply spectral prompt

        return Pspe, Y3





