# 檔案: source.py (最終版 - 無須載入權重)

import sys
import os
import torch
from thop import profile

# --- 將專案根目錄加入 Python 的搜尋路徑 ---
# 這對於載入您的自訂 Student 模型是必要的
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- 載入模型 ---
# 1. 使用 compressai.zoo 載入 Teacher 模型架構 (標準作法)
from compressai.zoo import image_models
# 2. 載入您自訂的 Student 模型
from onlyconvulution import SimpleConvStudentModel


def analyze_model(model, input_tensor):
    """
    分析給定模型的參數量、MACs 和 FLOPs。
    """
    # 將模型設為評估模式
    model.eval()

    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2
    params_m = params / 1_000_000
    macs_g = macs / 1_000_000_000
    flops_g = flops / 1_000_000_000

    print(f"  - 輸入尺寸 (Input Size): {input_tensor.shape}")
    print(f"  - 參數量 (Params): {params_m:.2f} M")
    print(f"  - GMACs (Giga MACs): {macs_g:.2f} G")
    print(f"  - GFLOPs (Giga FLOPs): {flops_g:.2f} G")
    print("-" * 30)


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 256, 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = dummy_input.to(device)

    # --- 分析 Teacher 模型 (TIC) ---
    print("分析 Teacher 模型 (TIC 架構)...")
    try:
        # ✨ 直接從 zoo 初始化模型架構，不載入任何權重 (pretrained=False)
        # 假設模型名為 'tic', 品質為 1
        teacher_model = image_models['tic'](quality=1, pretrained=False).to(device)
        analyze_model(teacher_model, dummy_input)
    except Exception as e:
        print(f"分析 Teacher 模型時出錯: {e}")
        print("請確認模型名稱 'tic' 已在 compressai.zoo 中正確註冊。")
        print("-" * 30)

    # --- 分析 Student 模型 (SimpleConvStudentModel) ---
    print("分析 Student 模型 (SimpleConvStudentModel 架構)...")
    try:
        student_model = SimpleConvStudentModel(N=128, M=196, teacher_channels=192).to(device)
        analyze_model(student_model, dummy_input)
    except Exception as e:
        print(f"分析 Student 模型時出錯: {e}")
        print("-" * 30)