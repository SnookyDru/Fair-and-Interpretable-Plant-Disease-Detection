import cv2
import numpy as np
import os

def load_rgb_hsi_images(main_dir):
    """
    inputs: main_dir: str, path to the main directory containing the dataset
    outputs: rgb_images: list of RGB images, hsi_images: list of HSI cubes
    """

    scenes_outer = sorted(os.listdir(main_dir))  # List all outer scenes (only applicable to CAVE dataset))

    rgb_images = []
    hsi_images = []

    for scene in scenes_outer:
        scene_path = os.path.join(main_dir, scene, scene)  # adjust if no inner folder
        if not os.path.isdir(scene_path):
            raise FileNotFoundError(f"Scene path not found: {scene_path}")

        # Get all files except .db
        files = [f for f in os.listdir(scene_path) if not f.lower().endswith(".db")]

        # Find RGB file (case-insensitive match)
        rgb_file = next((f for f in files if "_rgb" in f.lower() and f.lower().endswith(".png")), None)
        if rgb_file is None:
            raise FileNotFoundError(f"No RGB PNG found in {scene_path}")
        rgb = cv2.imread(os.path.join(scene_path, rgb_file))
        rgb_images.append(rgb)

        # Get and sort HSI band files
        band_files = [f for f in files if "_rgb" not in f.lower() and f.lower().endswith(".png")]
        band_files = sorted(band_files, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        if len(band_files) != 31:
            raise ValueError(f"{scene}: expected 31 HSI bands, found {len(band_files)}")

        # Load HSI bands
        hsi_bands = [cv2.imread(os.path.join(scene_path, bf), cv2.IMREAD_GRAYSCALE) for bf in band_files]
        hsi_cube = np.stack(hsi_bands, axis=-1)
        hsi_images.append(hsi_cube)

    # Validation
    if len(rgb_images) != 32 or len(hsi_images) != 32:
        raise ValueError(f"Expected 32 scenes, got {len(rgb_images)} RGB and {len(hsi_images)} HSI")
    
    print(f"Loaded {len(rgb_images)} RGB images and {len(hsi_images)} HSI cubes.")

    return rgb_images, hsi_images


def convert_hwb_to_bhw(hsi_list, rgb_list):
    hsi_list_bhw = [np.transpose(hsi, (2, 0, 1)) for hsi in hsi_list]
    rgb_list_bhw = [np.transpose(rgb, (2, 0, 1)) for rgb in rgb_list]

    return hsi_list_bhw, rgb_list_bhw