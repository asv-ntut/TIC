"""
ONNX Static Quantization (PTQ) for TIC

This script uses calibration data to perform static quantization.
This is much more accurate than dynamic quantization for sensitive models.

Usage:
    python quantize_static.py --img /path/to/calibration_img.tif
"""
import os
import sys
import argparse
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, quant_pre_process
from PIL import Image

class TICDataReader(CalibrationDataReader):
    def __init__(self, patches, input_name):
        self.patches = patches
        self.input_name = input_name
        self.datasize = len(patches)
        self.enum_data = iter([{self.input_name: p[np.newaxis, ...]} for p in patches])

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter([{self.input_name: p[np.newaxis, ...]} for p in self.patches])

def load_calibration_data(img_path, enc_path, hyper_path, num_patches=200, rgb_bands=(3, 2, 1)):
    """Load calibration patches from a file or a directory of images.
    
    Uses the same preprocessing as train.py (ImageFolder):
    - Sentinel-2 TIFF: Uses rgb_bands=(3, 2, 1) for B4, B3, B2 (true color)
    - Percentile-based normalization (p2, p98) for better contrast
    - Output: float32 in [0, 1] range
    
    Args:
        img_path: Path to image file or directory
        enc_path: Path to encoder ONNX
        hyper_path: Path to hyper decoder ONNX  
        num_patches: Number of patches to extract
        rgb_bands: Band indices for RGB (default: Sentinel-2 true color)
    """
    import glob
    
    # Try to import tifffile for satellite imagery
    try:
        import tifffile
        HAS_TIFFFILE = True
    except ImportError:
        HAS_TIFFFILE = False
        print("⚠️ tifffile not installed, satellite TIFF support limited")
    
    def normalize_to_float(img_array):
        """Normalize array to float32 [0, 1] range using percentile (same as train.py)."""
        if img_array.dtype == np.uint8:
            return img_array.astype(np.float32) / 255.0
        elif img_array.dtype in (np.uint16, np.int16):
            # 16-bit satellite: use percentile-based normalization
            p2, p98 = np.percentile(img_array, (2, 98))
            normalized = (img_array.astype(np.float32) - p2) / (p98 - p2 + 1e-8)
            return np.clip(normalized, 0, 1)
        elif img_array.dtype in (np.float32, np.float64):
            if img_array.max() <= 1.0:
                return img_array.astype(np.float32)
            else:
                p2, p98 = np.percentile(img_array, (2, 98))
                normalized = (img_array - p2) / (p98 - p2 + 1e-8)
                return np.clip(normalized, 0, 1).astype(np.float32)
        else:
            # Fallback: min-max normalization
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                return ((img_array - img_min) / (img_max - img_min)).astype(np.float32)
            return img_array.astype(np.float32)
    
    image_files = []
    if os.path.isdir(img_path):
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            image_files.extend(glob.glob(os.path.join(img_path, ext)))
    else:
        image_files = [img_path]

    if not image_files:
        raise ValueError(f"No images found at {img_path}")

    print(f"Loading {num_patches} patches from {len(image_files)} images...")
    print(f"  RGB bands: {rgb_bands} (B4=Red, B3=Green, B2=Blue for Sentinel-2)")
    print(f"  Normalization: Percentile (p2, p98) - same as train.py")
    
    patches = []
    patches_per_img = max(1, num_patches // len(image_files))
    
    for f in image_files:
        try:
            ext = os.path.splitext(f)[-1].lower()
            
            # Handle TIFF (potentially 16-bit satellite imagery)
            if ext in ['.tif', '.tiff'] and HAS_TIFFFILE:
                raw = tifffile.imread(f)
                
                # Handle dim order
                if raw.ndim == 2:
                    # Grayscale -> stack to 3 channels
                    raw = np.stack([raw, raw, raw], axis=-1)
                elif raw.ndim == 3:
                    # Check if channels-first (C, H, W) or channels-last (H, W, C)
                    if raw.shape[0] < raw.shape[2]:
                        raw = np.transpose(raw, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                
                # Extract RGB bands (matches train.py ImageFolder)
                num_bands = raw.shape[2]
                if rgb_bands is not None and num_bands >= max(rgb_bands) + 1:
                    r_idx, g_idx, b_idx = rgb_bands
                    img_array = raw[:, :, [r_idx, g_idx, b_idx]]
                else:
                    img_array = raw[:, :, :3]
                
                # Normalize using percentile (same as train.py)
                arr = normalize_to_float(img_array)
                arr = arr.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
                
            else:
                # Standard 8-bit image
                with Image.open(f) as img:
                    img = img.convert("RGB")
                    arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            
            c, h, w = arr.shape
            ps = 256
            count = 0
            # Sample patches from current image
            for y in range(0, h - ps, max(ps, h // 10)):
                for x in range(0, w - ps, max(ps, w // 10)):
                    patches.append(arr[:, y:y+ps, x:x+ps])
                    count += 1
                    if count >= patches_per_img: break
                if count >= patches_per_img: break
            
            if len(patches) >= num_patches: break
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    print(f"Collected total {len(patches)} image patches.")
    
    # Rest of the latent generation logic remains the same
    print("Running Encoder for calibration latents...")
    enc_sess = ort.InferenceSession(enc_path)
    y_latents = []
    z_latents = []
    for p in patches:
        out = enc_sess.run(None, {enc_sess.get_inputs()[0].name: p[np.newaxis, ...]})
        y_latents.append(out[0][0])
        z_latents.append(out[1][0])
    
    return {
        "tic_encoder": patches,
        "tic_hyper_decoder": z_latents,
        "tic_decoder": y_latents
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Image for calibration")
    args = parser.parse_args()

    print("=== ONNX Static Quantization (PTQ) ===")
    
    # Pre-check files
    required = ["tic_encoder.onnx", "tic_hyper_decoder.onnx", "tic_decoder.onnx"]
    for f in required:
        if not os.path.exists(f):
            print(f"Error: {f} not found.")
            return

    # Load calibration data for all stages
    calib_data = load_calibration_data(args.img, "tic_encoder.onnx", "tic_hyper_decoder.onnx")

    for m_base in ["tic_encoder", "tic_hyper_decoder", "tic_decoder"]:
        model_fp32 = f"{m_base}.onnx"
        model_pre = f"{m_base}_pre.onnx"
        model_int8 = f"{m_base}_static_int8.onnx"
        
        print(f"\nProcessing {model_fp32}...")
        
        # 1. Preprocess
        quant_pre_process(model_fp32, model_pre)
        
        # 2. Setup Data Reader
        sess = ort.InferenceSession(model_pre)
        input_name = sess.get_inputs()[0].name
        dr = TICDataReader(calib_data[m_base], input_name)
        
        # 3. Quantize
        quantize_static(
            model_input=model_pre,
            model_output=model_int8,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False
        )
        
        if os.path.exists(model_pre):
            os.remove(model_pre)
        print(f"✅ Generated {model_int8}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
