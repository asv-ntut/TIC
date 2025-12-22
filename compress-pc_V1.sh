#!/bin/bash
# ==============================================================================
# TIC PC Demo - Compression Script (V1)
# ==============================================================================

# 設定輸入檔案
INPUT_PATHS="4096_3072/hualien_RGB_Normalized_tile_r0_c0.tif 4096_3072/Taitung_RGB_Normalized_tile_r1_c0.tif"

# 設定模型與輸出路徑
CHECKPOINT="./1130stcheckpoint_best_loss.pth"
OUTPUT_DIR="output_satellite"

# 確保輸出目錄存在
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "   開始執行 TIC 壓縮模擬 (PC版 V1)"
echo "=================================================="
echo "模型路徑: $CHECKPOINT"
echo "輸出路徑: $OUTPUT_DIR"
echo "--------------------------------------------------"

# 執行 Python 壓縮程式
python onlycompress.py $INPUT_PATHS -p "$CHECKPOINT" -o "$OUTPUT_DIR"

echo ""
echo "=================================================="
echo "   壓縮作業完成"
echo "=================================================="
