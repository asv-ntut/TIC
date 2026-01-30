# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Try to import tifffile for multi-band TIFF support
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_sentinel2_tiff(filepath):
    """
    Load Sentinel-2 multi-band TIFF and convert to RGB.
    Uses B4 (Red), B3 (Green), B2 (Blue) bands with percentile normalization.
    
    Sentinel-2 band order (0-indexed):
    0: B1 (Coastal), 1: B2 (Blue), 2: B3 (Green), 3: B4 (Red),
    4: B5, 5: B6, 6: B7, 7: B8 (NIR), 8: B8A, 9: B9, 10: B10, 11: B11, 12: B12
    """
    # Read the TIFF file
    img_data = tifffile.imread(str(filepath))
    
    # Handle different array shapes
    if img_data.ndim == 3:
        # Check if bands are first or last dimension
        if img_data.shape[0] <= 13 and img_data.shape[0] < img_data.shape[1]:
            # Bands first: (C, H, W)
            bands = img_data
        else:
            # Bands last: (H, W, C)
            bands = np.transpose(img_data, (2, 0, 1))
    else:
        # Grayscale image
        bands = img_data[np.newaxis, :, :]
    
    num_bands = bands.shape[0]
    
    if num_bands >= 4:
        # Sentinel-2: B4 (Red), B3 (Green), B2 (Blue) -> indices 3, 2, 1
        rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
    elif num_bands == 3:
        # Already RGB
        rgb = np.transpose(bands, (1, 2, 0))
    else:
        # Grayscale, repeat to 3 channels
        rgb = np.repeat(bands[0][:, :, np.newaxis], 3, axis=2)
    
    # Percentile normalization (p2, p98) for better contrast
    rgb = rgb.astype(np.float32)
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    
    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    else:
        rgb = rgb / (rgb.max() + 1e-8)
    
    # Clip and convert to uint8
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(rgb)


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        filepath = self.samples[index]
        suffix = filepath.suffix.lower()
        
        # Handle multi-band TIFF files (Sentinel-2)
        if suffix in ('.tif', '.tiff') and HAS_TIFFFILE:
            try:
                img = load_sentinel2_tiff(filepath)
            except Exception:
                # Fallback to PIL if tifffile fails
                img = Image.open(filepath).convert("RGB")
        else:
            img = Image.open(filepath).convert("RGB")
        
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
