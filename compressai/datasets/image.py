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


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories.
    
    Supports both standard images (PNG, JPG) and multi-band TIFF files
    (e.g., Sentinel-2 satellite imagery with 13 bands).

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.tif
            - test/
                - img000.png
                - img001.tif

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
        rgb_bands (tuple): band indices to use as RGB for multi-band TIFF.
                          Default (3, 2, 1) for Sentinel-2 (B4=Red, B3=Green, B2=Blue).
                          Set to None to use first 3 bands.
    """

    def __init__(self, root, transform=None, split="train", rgb_bands=(3, 2, 1)):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.rgb_bands = rgb_bands

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        filepath = self.samples[index]
        suffix = filepath.suffix.lower()
        
        # Handle TIFF files (potentially multi-band)
        if suffix in ('.tif', '.tiff'):
            img = self._load_tiff(filepath)
        else:
            # Standard image formats (PNG, JPG, etc.)
            img = Image.open(filepath).convert("RGB")
        
        if self.transform:
            return self.transform(img)
        return img

    def _load_tiff(self, filepath):
        """Load a TIFF file, handling multi-band satellite imagery."""
        if HAS_TIFFFILE:
            # Use tifffile for multi-band support
            data = tifffile.imread(str(filepath))
            
            # Handle different array shapes
            if data.ndim == 2:
                # Grayscale image - convert to RGB by stacking
                img_array = np.stack([data, data, data], axis=-1)
            elif data.ndim == 3:
                # Check if channels-first (C, H, W) or channels-last (H, W, C)
                if data.shape[0] < data.shape[2]:
                    # Likely channels-first: (C, H, W) -> (H, W, C)
                    data = np.transpose(data, (1, 2, 0))
                
                num_bands = data.shape[2]
                
                if num_bands >= 3:
                    # Multi-band: extract RGB bands
                    if self.rgb_bands is not None:
                        r_idx, g_idx, b_idx = self.rgb_bands
                        # Clamp to valid range
                        r_idx = min(r_idx, num_bands - 1)
                        g_idx = min(g_idx, num_bands - 1)
                        b_idx = min(b_idx, num_bands - 1)
                    else:
                        # Use first 3 bands
                        r_idx, g_idx, b_idx = 0, 1, 2
                    
                    img_array = data[:, :, [r_idx, g_idx, b_idx]]
                elif num_bands == 1:
                    # Single band - convert to RGB
                    img_array = np.concatenate([data, data, data], axis=-1)
                else:
                    # 2 bands - add a third
                    img_array = np.concatenate([data, data[:, :, :1]], axis=-1)
            else:
                raise ValueError(f"Unexpected TIFF array shape: {data.shape}")
            
            # Normalize to 0-255 uint8 range
            img_array = self._normalize_to_uint8(img_array)
            
            return Image.fromarray(img_array, mode='RGB')
        else:
            # Fallback: try PIL (may fail for multi-band TIFF)
            try:
                return Image.open(filepath).convert("RGB")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot open TIFF file {filepath}. "
                    f"Install tifffile for multi-band TIFF support: pip install tifffile"
                ) from e

    def _normalize_to_uint8(self, img_array):
        """Normalize array to uint8 (0-255) range."""
        if img_array.dtype == np.uint8:
            return img_array
        elif img_array.dtype in (np.float32, np.float64):
            # Float: assume 0-1 range
            if img_array.max() <= 1.0:
                return (img_array * 255).clip(0, 255).astype(np.uint8)
            else:
                # Normalize using percentile for better contrast
                p2, p98 = np.percentile(img_array, (2, 98))
                img_array = (img_array - p2) / (p98 - p2 + 1e-8) * 255
                return img_array.clip(0, 255).astype(np.uint8)
        elif img_array.dtype == np.uint16:
            # 16-bit: use percentile-based normalization for satellite imagery
            p2, p98 = np.percentile(img_array, (2, 98))
            img_array = (img_array - p2) / (p98 - p2 + 1e-8) * 255
            return img_array.clip(0, 255).astype(np.uint8)
        else:
            # Other types: simple min-max normalization
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min) * 255
            return img_array.clip(0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.samples)
