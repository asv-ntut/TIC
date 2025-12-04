import os
import cv2
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

# RGB mean/std for normalization
S2RGB_MEAN = [100.708, 87.489, 61.932]
S2RGB_STD = [68.550, 47.647, 40.592]

def normalize_s2rgb(image):
    """Normalize and scale RGB bands for display."""
    image = (image - np.array(S2RGB_MEAN).reshape(-1,1,1)) / np.array(S2RGB_STD).reshape(-1,1,1)
    image = image * 40 + 127.5  # stretch contrast
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def process_zarr_directory(source_dir: Path, output_dir: Path):
    """
    Convert all .zarr/.zarr.zip files under source_dir into 256x256 RGB PNG images under output_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.zarr') or f.endswith('.zarr.zip')])

    for file in tqdm(files, desc=f"Processing {source_dir.name}"):
        try:
            file_path = source_dir / file
            ds = xr.open_zarr(file_path)

            # Get RGB data: select time=0
            bands = ds.bands
            if 'time' in bands.dims:
                bands = bands.isel(time=0)

            rgb = bands.values

            # Auto-shape handling
            if rgb.ndim == 4:  # (T, C, H, W)
                rgb = rgb[0, :3]
            elif rgb.ndim == 3:
                if rgb.shape[0] >= 3:
                    rgb = rgb[:3]
                elif rgb.shape[-1] >= 3:
                    rgb = np.transpose(rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                else:
                    print(f"⚠️ Too few channels in {file}, skipping.")
                    continue
            else:
                print(f"⚠️ Unexpected shape {rgb.shape} in {file}, skipping.")
                continue

            # Normalize and reshape
            rgb = normalize_s2rgb(rgb)       # (3, H, W)
            rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, C)
            rgb_resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)

            # Save to output
            out_name = Path(file).stem + '.png'
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")
            continue

# === 主流程 ===

# 資料來源與輸出結構
base_source = Path(r"D:\ssl4eo-s12\ssl4eo-s12")  # 含 train/val/test 各 S2RGB 資料夾
base_output = Path(r"C:\Users\Matt\Desktop\256 256 3")

for split in ['train', 'val', 'test']:
    source_folder = base_source / split / 'S2RGB'
    output_folder = base_output / split
    process_zarr_directory(source_folder, output_folder)

print("✅ 所有轉換完成")