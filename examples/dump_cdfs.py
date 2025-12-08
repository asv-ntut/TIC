import torch
import sys
import os
from conv2 import SimpleConvStudentModel, get_scale_table

def dump_cdfs():
    checkpoint_path = "../1130stcheckpoint_best_loss.pth"
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    N, M = 128, 192
    try:
        N = new_state_dict['g_a.0.weight'].size(0)
        keys = sorted([k for k in new_state_dict.keys() if 'g_a' in k and 'weight' in k])
        M = new_state_dict[keys[-1]].size(0)
    except:
        pass

    # [FIX] Remap keys for CompressAI version mismatch
    for k in list(new_state_dict.keys()):
        if "entropy_bottleneck._matrix" in k:
            idx = k.split("_matrix")[-1]
            new_k = k.replace(f"_matrix{idx}", f"matrices.{idx}")
            new_state_dict[new_k] = new_state_dict.pop(k)
        elif "entropy_bottleneck._bias" in k:
            idx = k.split("_bias")[-1]
            new_k = k.replace(f"_bias{idx}", f"biases.{idx}")
            new_state_dict[new_k] = new_state_dict.pop(k)
        elif "entropy_bottleneck._factor" in k:
            idx = k.split("_factor")[-1]
            new_k = k.replace(f"_factor{idx}", f"factors.{idx}")
            new_state_dict[new_k] = new_state_dict.pop(k)

    model = SimpleConvStudentModel(N=N, M=M)
    model.load_state_dict(new_state_dict, strict=False)
    
    # [FIX] Force CPU execution for CDF generation to match new runtime flow
    # This ensures the probability tables are bit-exact with the CPU decoding path
    model = model.to('cpu')
    
    # Force update to generate CDFs (on CPU)
    model.update(force=True)
    
    eb = model.entropy_bottleneck
    
    medians = eb._get_medians().detach()
    
    with open("fixed_cdfs.py", "w") as f:
        f.write(f"FIXED_EB_CDF = {eb._quantized_cdf.tolist()}\n")
        f.write(f"FIXED_EB_OFFSET = {eb._offset.tolist()}\n")
        f.write(f"FIXED_EB_LENGTH = {eb._cdf_length.tolist()}\n")
        f.write(f"FIXED_EB_MEDIANS = {medians.tolist()}\n")
    print("Successfully wrote fixed_cdfs.py")

if __name__ == "__main__":
    dump_cdfs()
