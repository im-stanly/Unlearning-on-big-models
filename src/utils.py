import numpy as np
from pathlib import Path
import os
from PIL import Image


def load_images(dir_path: str) -> list[np.ndarray]:
    directory_path = Path(dir_path)
    file_names = os.listdir(directory_path)
    images = []
    for file_name in sorted(file_names):
        file_path = directory_path / file_name
        loaded_image = Image.open(file_path)
        images.append(np.asarray(loaded_image))

    return images
