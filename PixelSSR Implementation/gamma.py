import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import torch

def fit_gamma_distribution(hsi_list_bhw,
                       force_loc0=True,
                       remove_zeros=True,
                       max_samples=None,
                       normalize=True,
                       verbose=True,
                       rng_seed=0):
    """
    Fit a single Gamma distribution to the entire HxWxC hyperspectral dataset
    (full-cube flattening across scenes and bands).

    Args:
        hsi_list_bhw (list of np.array): list of HSI cubes in (B, H, W) format.
        force_loc0 (bool): if True, fit with loc=0 (floc=0).
        remove_zeros (bool): if True, drop zero values before fitting.
        max_samples (int or None): if not None, randomly subsample up to this many voxels for fitting.
        normalize (bool): if True and data appears integer/ >1, divide by 65535.0 (common for 16-bit HSI).
        verbose (bool): print summary info.
        rng_seed (int): random seed for subsampling.

    Returns:
        params (dict): {"a": shape, "loc": loc, "scale": scale}
        fit_data (np.ndarray): the 1D array used for fitting (after preprocessing) -- returned for plotting/debugging
    """
    rng = np.random.RandomState(rng_seed)

    # Collect all voxel values (flatten across scenes and bands)
    all_vals = []
    for cube in hsi_list_bhw:
        arr = np.asarray(cube)
        if arr.ndim != 3:
            raise ValueError("Each HSI cube must be 3D (B,H,W).")
        all_vals.append(arr.ravel())
    all_vals = np.concatenate(all_vals).astype(np.float64)

    # normalization: if data is in integer range or max > 1, scale down
    if normalize and all_vals.size > 0:
        maxv = np.nanmax(all_vals)
        if maxv > 1.0:
            # assume 16-bit reflectance -> scale to [0,1]
            scale_factor = 65535.0
            if maxv < 257:   # maybe 8-bit images (0-255)
                scale_factor = 255.0
            all_vals = all_vals / float(scale_factor)
            if verbose:
                print(f"[fit_gamma_fullcube] Normalized values by {scale_factor} (max before scale = {maxv:.2f}).")

    # Remove NaNs / negatives; optionally remove zeros
    all_vals = all_vals[np.isfinite(all_vals)]
    if remove_zeros:
        all_vals = all_vals[all_vals > 0.0]
    else:
        # clamp negative to 0 (and tiny positives)
        all_vals = np.clip(all_vals, 1e-12, None)

    if all_vals.size == 0:
        raise RuntimeError("No valid positive samples found after preprocessing.")

    # Subsample for speed if requested
    if max_samples is not None and all_vals.size > max_samples:
        idx = rng.choice(all_vals.size, size=max_samples, replace=False)
        fit_vals = all_vals[idx]
        if verbose:
            print(f"[fit_gamma_fullcube] Subsampled from {all_vals.size} -> {fit_vals.size} voxels for fitting.")
    else:
        fit_vals = all_vals
        if verbose:
            print(f"[fit_gamma_fullcube] Using {fit_vals.size} voxels for fitting.")

    # Fit Gamma using MLE (SciPy). Force loc=0 if requested.
    if force_loc0:
        a, loc, scale = gamma.fit(fit_vals, floc=0)
    else:
        a, loc, scale = gamma.fit(fit_vals)

    params = {"a": float(a), "loc": float(loc), "scale": float(scale)}
    if verbose:
        print("[fit_gamma_fullcube] Fitted params:", params)

    return params, fit_vals



def generate_gamma_cube(h, w, b, params, seed=0, normalize=True, dtype=torch.float32):
    """
    Generate a full synthetic Gamma-modeled hyperspectral cube.
    
    Args:
        h (int): Height of cube.
        w (int): Width of cube.
        b (int): Number of spectral bands.
        params (dict): {'a': shape, 'loc': loc, 'scale': scale} from fit_gamma_fullcube.
        seed (int): Random seed for reproducibility.
        normalize (bool): If True, clip values to [0,1].
        dtype: torch data type for output.

    Returns:
        torch.Tensor: Cube of shape (B, H, W) in channel-first format.
    """
    rng = np.random.default_rng(seed)

    # Generate full cube in (H, W, B) order
    cube = rng.gamma(shape=params["a"],
                     scale=params["scale"],
                     size=(h, w, b)) + params["loc"]

    # Optional normalization to [0,1]
    if normalize:
        cube_min, cube_max = cube.min(), cube.max()
        if cube_max > cube_min:
            cube = (cube - cube_min) / (cube_max - cube_min)

    # Convert to (B, H, W) and torch.Tensor
    cube_ch_first = np.transpose(cube, (2, 0, 1))

    print("Gamma-modeled cube shape:", cube_ch_first.shape)
    print("Min/Max:", cube_ch_first.min().item(), cube_ch_first.max().item())
    return torch.tensor(cube_ch_first, dtype=dtype)



def apply_gamma_mask(cube, ratio=0.01, seed=0):
    """
    Apply a point spectra mask to the Gamma-modeled cube.
    
    Args:
        cube (torch.Tensor or np.ndarray): shape (B, H, W) - Gamma-modeled cube
        ratio (float): ratio of pixels to KEEP as point spectra (e.g., 0.0001 for 0.01%)
        seed (int): random seed for reproducibility
    
    Returns:
        masked_cube (torch.Tensor): same shape as cube, masked values
        mask (torch.Tensor): binary mask (1=keep, 0=mask out), same shape as cube
    """
    if isinstance(cube, torch.Tensor):
        cube_np = cube.cpu().numpy()
    else:
        cube_np = np.array(cube)

    B, H, W = cube_np.shape
    rng = np.random.default_rng(seed)

    # Number of pixels to keep (same spatial locations for all bands)
    total_pixels = H * W
    keep_pixels = int(np.floor(ratio * total_pixels))

    # Create flat mask for spatial positions
    spatial_mask = np.zeros(total_pixels, dtype=bool)
    spatial_mask[rng.choice(total_pixels, size=keep_pixels, replace=False)] = True
    spatial_mask = spatial_mask.reshape(H, W)

    # Broadcast mask to all bands
    mask_full = np.broadcast_to(spatial_mask, (B, H, W))

    # Apply mask
    masked_cube = cube_np * mask_full

    # Convert back to torch
    masked_cube_torch = torch.tensor(masked_cube, dtype=torch.float32)
    mask_torch = torch.tensor(mask_full.astype(np.float32), dtype=torch.float32)

    print("Masked cube shape:", masked_cube_torch.shape)
    print("Mask shape:", mask_torch.shape)
    print("Non-zero points in mask:", mask_torch.sum().item())

    return masked_cube_torch, mask_torch




