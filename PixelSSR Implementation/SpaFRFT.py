import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_frft.frft_module import frft, ifrft

# --------------------------
# Utilities: scanning index creation
# --------------------------
def make_scan_indices(H, W, mode):
    coords = []
    if mode == 'row':
        for r in range(H):
            for c in range(W):
                coords.append((r, c))
    elif mode == 'col':
        for c in range(W):
            for r in range(H):
                coords.append((r, c))
    elif mode == 'diag':
        for s in range(H + W - 1):
            for r in range(H):
                c = s - r
                if 0 <= c < W:
                    coords.append((r, c))
    elif mode == 'diag_rev':
        for s in range(H + W - 2, -1, -1):
            for r in range(H):
                c = s - r
                if 0 <= c < W:
                    coords.append((r, c))
    else:
        raise ValueError("unknown mode")
    return coords

def coords_to_rowmajor_idx(coords, W):
    return [r * W + c for (r, c) in coords]

def inverse_permutation(idx_list):
    L = len(idx_list)
    inv = [0] * L
    for j, val in enumerate(idx_list):
        inv[val] = j
    return inv

# --------------------------
# Sin-Cos Positional Encoding
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, J, D):
        super().__init__()
        self.register_buffer("pos_emb", self._build_positional_encoding(J, D), persistent=False)

    def _build_positional_encoding(self, J, D):
        pos = torch.arange(J, dtype=torch.float).unsqueeze(1)   # (J,1)
        i = torch.arange(D, dtype=torch.float).unsqueeze(0)     # (1,D)
        div_term = torch.exp(-math.log(10000.0) * (2 * (i // 2) / D))
        pe = torch.zeros(J, D)
        pe[:, 0::2] = torch.sin(pos * div_term[:, 0::2])
        pe[:, 1::2] = torch.cos(pos * div_term[:, 1::2])
        return pe.unsqueeze(0)  # (1,J,D)

    def forward(self, x):  # (B,J,D)
        return x + self.pos_emb.to(x.device)

# --------------------------
# Vision Mamba VIM block (forward/backward selective SSM)
# --------------------------
class VimBlock(nn.Module):
    """
    Input:  tokens (B, J, D)
    Output: tokens (B, J, D)
    Hyper-params:
      D: token hidden width
      E: expanded width for x/z streams
      N: SSM state size
    """
    def __init__(self, D, E=384, N=16, conv_ks=3):
        super().__init__()
        self.D, self.E, self.N = D, E, N

        # Pre-norm
        self.norm = nn.LayerNorm(D)

        # Two linear projections -> x, z  (both shape E)
        self.to_x = nn.Linear(D, E)
        self.to_z = nn.Linear(D, E)

        # Depthwise Conv1d along sequence (forward/backward)
        padding = conv_ks // 2
        self.conv_fwd = nn.Conv1d(E, E, kernel_size=conv_ks, padding=padding, groups=E, bias=True)
        self.conv_bwd = nn.Conv1d(E, E, kernel_size=conv_ks, padding=padding, groups=E, bias=True)

        # Generate per-token SSM parameters (B_t, C_t, Δ_t) in state space N
        self.gen_B_f = nn.Linear(E, N)
        self.gen_C_f = nn.Linear(E, N)
        self.gen_D_f = nn.Linear(E, N)

        self.gen_B_b = nn.Linear(E, N)
        self.gen_C_b = nn.Linear(E, N)
        self.gen_D_b = nn.Linear(E, N)

        # Learnable diagonal A (kept stable via -softplus)
        self.A_log = nn.Parameter(torch.zeros(N))  # (N,)

        # Project token content (E) -> state space (N) for driving input u_t
        self.in_to_state = nn.Linear(E, N)

        # Readout: state (N) -> E, then to D
        self.readout = nn.Linear(N, E)
        self.proj_out = nn.Linear(E, D)

        # Gate activation for z
        self.act = nn.SiLU()

    @torch.no_grad()
    def _stable_A(self):
        # Negative diagonal for stability
        return -F.softplus(self.A_log).view(1, 1, -1)  # (1,1,N) broadcastable

    def _discretize(self, B, Delta):
        """
        ZOH discretization for diagonal A:
          A_bar = exp(Delta * A)
          B_bar = (A_bar - 1) / A * B     (elementwise)
        Shapes:
          B, Delta: (B,J,N)
        Returns:
          A_bar, B_bar: (B,J,N)
        """
        A = self._stable_A()               # (1,1,N)
        A_bar = torch.exp(Delta * A)       # (B,J,N)  broadcasting
        eps = 1e-6
        factor = (A_bar - 1.0) / (A + eps) # (B,J,N)
        B_bar = factor * B                 # (B,J,N)
        return A_bar, B_bar

    def _scan_direction(self, x_tokens, conv, gen_B, gen_C, gen_D):
        """
        One directional SSM pass.
        Args:
          x_tokens: (B,J,E)
          conv: depthwise conv1d module
          gen_B/C/D: linear generators (E->N)
        Returns:
          y: (B,J,E)
        """
        Bsz, J, E = x_tokens.shape

        # Depthwise conv along sequence: (B,E,J) -> (B,E,J) -> (B,J,E)
        x_conv = conv(x_tokens.transpose(1, 2)).transpose(1, 2)  # (B,J,E)

        # Per-token SSM parameters in state space
        B_t = gen_B(x_conv)                 # (B,J,N)
        C_t = gen_C(x_conv)                 # (B,J,N)
        Delta_t = F.softplus(gen_D(x_conv)) # (B,J,N), positive

        # Discretize A,B with ZOH
        A_bar, B_bar = self._discretize(B_t, Delta_t)  # (B,J,N)

        # Recurrent scan over sequence
        h = torch.zeros(Bsz, self.N, device=x_tokens.device, dtype=x_tokens.dtype)  # (B,N)
        Ys = []
        for t in range(J):
            # Project token content into state space as driving input u_t (B,N)
            u_t = self.in_to_state(x_conv[:, t, :])    # (B,N)
            # State update (elementwise on N)
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * u_t  # (B,N)
            # Readout: (h ⊙ C_t) -> E
            y_t = self.readout(h * C_t[:, t, :])            # (B,E)
            Ys.append(y_t)

        y = torch.stack(Ys, dim=1)  # (B,J,E)
        return y

    def forward(self, tokens):  # (B,J,D)
        # Pre-norm
        t = self.norm(tokens)                  # (B,J,D)

        # Two streams: x (content) and z (gate)
        x = self.to_x(t)                       # (B,J,E)
        z = self.to_z(t)                       # (B,J,E)
        gate = torch.sigmoid(self.act(z))      # (B,J,E)

        # Forward and backward SSM passes
        y_f = self._scan_direction(x, self.conv_fwd, self.gen_B_f, self.gen_C_f, self.gen_D_f)  # (B,J,E)

        x_rev = torch.flip(x, dims=[1])
        y_b_rev = self._scan_direction(x_rev, self.conv_bwd, self.gen_B_b, self.gen_C_b, self.gen_D_b)
        y_b = torch.flip(y_b_rev, dims=[1])    # (B,J,E)

        # Gate + fuse, project back to D, residual add
        y = (y_f + y_b) * gate                 # (B,J,E)
        out = self.proj_out(y)                 # (B,J,D)
        return tokens + out

# --------------------------
# MScanBlock (ViM-style patch tokenization, fusion AFTER folding)
# --------------------------
class MScanBlock(nn.Module):
    def __init__(self, in_ch=40, patch_size=16, embed_dim=256, H=512, W=512,
                 paths=('row','col','diag','diag_rev'),
                 vim_E=384, vim_N=16, vim_ks=3):
        super().__init__()
        self.in_ch = in_ch
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.H, self.W = H, W
        self.paths = paths
        self.num_patches = (H // patch_size) * (W // patch_size)

        # Patch vector (P²·C) → D
        self.proj_in = nn.Linear(patch_size * patch_size * in_ch, embed_dim)

        # Positional encoding (sin–cos)
        self.pos_embed = PositionalEncoding(self.num_patches, embed_dim)

        # Vim blocks per path
        self.vims = nn.ModuleList([
            VimBlock(D=embed_dim, E=vim_E, N=vim_N, conv_ks=vim_ks)
            for _ in range(len(paths))
        ])

        # D → patch space, then fold
        self.proj_out = nn.Linear(embed_dim, patch_size * patch_size * in_ch)

        # Fuse full images from all paths
        self.fuse_conv = nn.Conv2d(len(paths) * in_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, feat):  # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        P = self.patch_size
        hP, wP = H // P, W // P
        assert C == self.in_ch
        assert H % P == 0 and W % P == 0

        # Patchify → (B, J, C*P*P)
        patches = F.unfold(feat, kernel_size=P, stride=P).transpose(1, 2)  # (B,J,CP2)

        # Project to D and add positional encoding
        tokens = self.proj_in(patches)          # (B,J,D)
        tokens = self.pos_embed(tokens)         # (B,J,D)

        path_imgs = []
        for i, mode in enumerate(self.paths):
            coords = make_scan_indices(hP, wP, mode)
            idx = coords_to_rowmajor_idx(coords, wP)
            inv_idx = inverse_permutation(idx)

            # Reorder by scan, Vim, restore
            tokens_scan = tokens[:, idx, :]             # (B,J,D)
            tokens_out = self.vims[i](tokens_scan)      # (B,J,D)
            tokens_out = tokens_out[:, inv_idx, :]      # (B,J,D)

            # Project back to patch space and fold to image
            patch_feats = self.proj_out(tokens_out).transpose(1, 2)  # (B, C*P*P, J)
            img_path = F.fold(patch_feats, output_size=(H, W), kernel_size=P, stride=P)  # (B,C,H,W)
            path_imgs.append(img_path)

        # Fuse full-path images
        concat_img = torch.cat(path_imgs, dim=1)        # (B, len(paths)*C, H, W)
        out = self.fuse_conv(concat_img)                # (B, C, H, W)
        return out

# --------------------------
# FRFT modules
# --------------------------
class MFrFT(nn.Module):
    def __init__(self, a0=0.5, a1=0.5):
        super().__init__()
        self.a0 = a0
        self.a1 = a1

    def forward(self, x):
        # Apply 2D FRFT: first along height, then width
        return frft(frft(x, self.a0, dim=-2), self.a1, dim=-1)

class MiFrFT(nn.Module):
    def __init__(self, a0=0.5, a1=0.5):
        super().__init__()
        self.a0 = a0
        self.a1 = a1

    def forward(self, x):
        return torch.abs(ifrft(ifrft(x, self.a0, dim=-2), self.a1, dim=-1))

# --------------------------
# Wrapper:
# F1 = M3×3(X),FSpa = Mscan(F1)
# FFRFT = MiFRFT (Mscan(MFRFT (F1)))
# Pspa = M3×3(Mctc(Fspa,FFRFT ) ·MSiLU (F1)
# --------------------------
class SpaFRFT(nn.Module):
    def __init__(self, in_ch=40, C=40, patch_size=16, embed_dim=256, H=512, W=512,
                 vim_E=384, vim_N=16, vim_ks=3, frft_a0=0.5, frft_a1=0.5):
        super().__init__()
        self.scan = MScanBlock(in_ch=C, patch_size=patch_size, embed_dim=embed_dim,
                                         H=H, W=W, vim_E=vim_E, vim_N=vim_N, vim_ks=vim_ks)
        self.frft = MFrFT(a0=frft_a0, a1=frft_a1)
        self.ifrft = MiFrFT(a0=frft_a0, a1=frft_a1)
        self.scanFRFT = MScanBlock(in_ch=C, patch_size=patch_size, embed_dim=embed_dim,
                                          H=H, W=W, vim_E=vim_E, vim_N=vim_N, vim_ks=vim_ks)
        self.fused_conv = nn.Conv2d(2 * C, C, kernel_size=3, padding=1)
        self.MSilu = nn.SiLU()
        self.conv_out = nn.Conv2d(C, 3, kernel_size=3, padding=1)  # 40 channels to 3 RGB channels

    def forward(self, F1):
        FSpa = self.scan(F1)

        # FRFT branch
        Fracft = self.frft(F1)               # FRFT domain
        F_real = Fracft.real
        F_imag = Fracft.imag
        Ff_r = self.scanFRFT(F_real)   # scan in FRFT domain
        Ff_i = self.scanFRFT(F_imag)   # scan in FRFT domain
        Ff_s = torch.complex(Ff_r, Ff_i)  # combine real and imaginary parts
        FFRFT = self.ifrft(Ff_s)         # back to spatial domain

        # Combine and activate
        cat = torch.cat([FSpa, FFRFT], dim=1)      # (B, 2C, H, W)
        fused = self.fused_conv(cat)
        silu = self.MSilu(fused * F1)          # elementwise multiply with SiLU(F1)
        Pspa = self.conv_out(silu)
        return Pspa # (B, 3, H, W)
