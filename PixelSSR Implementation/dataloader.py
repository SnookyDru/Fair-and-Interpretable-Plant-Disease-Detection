import torch
from torch.utils.data import Dataset, DataLoader

class RGBHSIDataset(Dataset):
    def __init__(self, rgb_list, hsi_list):
        """
        rgb_list: list of numpy arrays with shape (3, H, W)
        hsi_list: list of numpy arrays with shape (B, H, W)
        """
        assert len(rgb_list) == len(hsi_list), "RGB and HSI lists must have the same length"
        self.rgb_list = [img / 255.0 for img in rgb_list]  # Normalize to [0, 1]
        self.hsi_list = [cube / 255.0 for cube in hsi_list]  # Normalize to [0, 1]

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        rgb = torch.tensor(self.rgb_list[idx], dtype=torch.float32)
        hsi = torch.tensor(self.hsi_list[idx], dtype=torch.float32)
        return rgb, hsi

def create_dataloader(rgb_list_bhw, hsi_list_bhw, batch_size=1, num_workers=0):
    """
    Create a DataLoader for the RGB-HSI dataset.
    
    rgb_list_bhw: list of RGB images in BHW format
    hsi_list_bhw: list of HSI cubes in BHW format
    batch_size: int, size of each batch
    """
    dataset = RGBHSIDataset(rgb_list_bhw, hsi_list_bhw)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # num_workers=0 for Windows
    print(f"Created DataLoader with {len(dataset)} samples, batch size {batch_size}, num workers {num_workers}.")
    return dataset, dataloader


