import os
import cv2
import numpy as np

def build_cataract_heightmap(image_path, save_folder, prefix="cataract"):
    """
    Generates a pseudo 3D heightmap for cataract.
    Based on the green-channel brightness + Gaussian blur.
    """
    os.makedirs(save_folder, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")

    # convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Cataract → affects clarity → use brightness as depth
    green = rgb[:, :, 1]

    # smooth for 3D surface
    height_map = cv2.GaussianBlur(green, (21, 21), 0)

    # normalize heightmap
    height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)

    heightmap_name = f"{prefix}_heightmap.png"
    heightmap_path = os.path.join(save_folder, heightmap_name)

    cv2.imwrite(heightmap_path, height_map)

    return os.path.basename(heightmap_path)
