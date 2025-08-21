import torch
import torch.nn as nn
from DRPN import DRPN
from DyProMamba import DyProMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PixelSSR(nn.Module):
    def __init__(self, out_bands=31):
        super().__init__()
        self.out_bands = out_bands

        self.DRPN_model = DRPN(in_ch=3, C=40, patch_size=16, embed_dim=256, H=512, W=512,
                 vim_E=384, vim_N=16, vim_ks=3, frft_a0=0.5, frft_a1=0.5, out_bands=self.out_bands).to(device)
        
        self.DyProMamba_model = DyProMamba(C_hsi=self.out_bands, in_ch_x=3, H=512, W=512,
                             patch_size=16, embed_dim=256, vim_E=384, vim_N=16, vim_ks=3)

    def forward(self, x, masked_cube):

        Pspe, Y3 = self.DRPN_model(x.to(device), masked_cube.to(device))
        Yr, Y4, Y5 = self.DyProMamba_model(x.to(device), Y3.to(device), Pspe.to(device))

        return Yr, Y3