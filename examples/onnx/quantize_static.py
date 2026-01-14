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

def load_calibration_data(img_path, enc_path, hyper_path, num_patches=200):
    """Load calibration patches from a file or a directory of images"""
    import glob
    
    image_files = []
    if os.path.isdir(img_path):
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            image_files.extend(glob.glob(os.path.join(img_path, ext)))
    else:
        image_files = [img_path]

    if not image_files:
        raise ValueError(f"No images found at {img_path}")

    print(f"Loading {num_patches} patches from {len(image_files)} images...")
    
    patches = []
    patches_per_img = max(1, num_patches // len(image_files))
    
    for f in image_files:
        try:
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
