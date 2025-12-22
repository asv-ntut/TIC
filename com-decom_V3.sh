#!/bin/bash
# ==============================================================================
# 衛星影像 壓縮/解壓縮 整合測試腳本 (PC Loopback Test) V3
# 功能：執行完整的 End-to-End 流程：原始影像 -> 壓縮 -> 比對統計 -> 解壓縮還原
# V3 更新：
# 1. 修正路徑為當前目錄 (./)
# 2. 移除自動修改 Python 檔的邏輯，維持原本小數位數 (.4f)
# 3. 使用 python -u 強制單行更新流暢顯示
# ==============================================================================

# ==============================================================================
# 【設定區】請在這裡修改變數
# ==============================================================================

# 1. 輸入影像路徑 (原始影像)
INPUT_PATHS="4096_3072/hualien_RGB_Normalized_tile_r0_c0.tif 4096_3072/Taitung_RGB_Normalized_tile_r1_c0.tif"

# 2. 模型 Checkpoint 路徑 (當前目錄)
CHECKPOINT="./1130stcheckpoint_best_loss.pth"

# 3. 輸出設定
COMPRESS_OUTPUT_DIR="output_satellite"  # 壓縮後的 bitstream 存放資料夾
DECOMPRESS_OUTPUT_DIR="recon_satellite" # 解壓縮後的重建影像存放資料夾

# 4. Python 腳本名稱
PY_COMPRESS="onlycompress.py"
PY_DECOMPRESS="onlydecompress.py"

# ==============================================================================
# 以下為腳本邏輯
# ==============================================================================

# --- 顏色定義 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_separator() { echo "============================================================"; }

# --- 環境檢查 ---
if [[ -z "$CONDA_DEFAULT_ENV" && -z "$VIRTUAL_ENV" ]]; then
    log_warning "未檢測到虛擬環境，建議先執行 conda activate tic10"
else
    log_info "目前環境: $CONDA_DEFAULT_ENV (或 $VIRTUAL_ENV)"
fi

# --- 開始處理 ---
echo ""
echo -e "${CYAN}"
cat << 'EOF'
      /--|  |  |--\
 |-------| (O) |-------|       AI 影像壓縮/解壓縮
 |-------| (O) |-------|         Loopback Test
 |-------| (O) |-------|     ━━━━━━━━━━━━━━━━━━━━
      \--|__|__|--/              (V3: Simple)
         |     |        
EOF
echo -e "${NC}"
print_separator
log_info "開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Checkpoint: $CHECKPOINT"
log_info "輸入路徑: $INPUT_PATHS"
print_separator

SCRIPT_START=$(date +%s)

# ==============================================================================
# 階段一：壓縮 (Compression)
# ==============================================================================
log_step "階段 1/2: 執行壓縮..."

mkdir -p "$COMPRESS_OUTPUT_DIR"
read -ra PATH_ARRAY <<< "$INPUT_PATHS"

# 檢查輸入檔案是否存在
for f in "${PATH_ARRAY[@]}"; do
    if [ ! -f "$f" ]; then
        log_error "找不到輸入檔案: $f"
        log_error "請確認 4096_3072 資料夾內是否有該影像"
        exit 1
    fi
done

# 執行 Python 壓縮
# 使用 -u (unbuffered) 確保 python 裡的 end='\r' 能即時刷新螢幕
python -u "$PY_COMPRESS" "${PATH_ARRAY[@]}" -p "$CHECKPOINT" -o "$COMPRESS_OUTPUT_DIR"
COMPRESS_STATUS=$?

if [ $COMPRESS_STATUS -ne 0 ]; then
    log_error "壓縮過程發生錯誤 (exit code: $COMPRESS_STATUS)"
    exit 1
fi

echo ""
log_success "壓縮階段完成。"

# --- 計算壓縮統計 ---
print_separator
log_info "壓縮數據統計:"
echo ""

for input_path in "${PATH_ARRAY[@]}"; do
    filename=$(basename "$input_path")
    basename_no_ext="${filename%.*}"
    compressed_folder="${COMPRESS_OUTPUT_DIR}/${basename_no_ext}"
    
    if [ -f "$input_path" ] && [ -d "$compressed_folder" ]; then
        original_size=$(cat "$input_path" | wc -c)
        compressed_size=$(cat "$compressed_folder"/* | wc -c)
        
        if [ "$original_size" -gt 0 ]; then
            reduction=$(echo "scale=1; 100 - 100 * $compressed_size / $original_size" | bc)
            ratio=$(echo "scale=2; $original_size / $compressed_size" | bc)
        else
            reduction="N/A"
            ratio="N/A"
        fi
        
        echo -e "  ${BLUE}${filename}${NC}"
        echo -e "    原始: $(numfmt --to=iec-i --suffix=B $original_size) → 壓縮: $(numfmt --to=iec-i --suffix=B $compressed_size)"
        echo -e "    節省空間: ${GREEN}${reduction}%${NC} (壓縮比 ${ratio}:1)"
    fi
done

print_separator

# ==============================================================================
# 階段二：解壓縮 (Decompression)
# ==============================================================================
log_step "階段 2/2: 執行解壓縮與還原..."

mkdir -p "$DECOMPRESS_OUTPUT_DIR"
DECOMP_COUNT=0

for input_path in "${PATH_ARRAY[@]}"; do
    filename=$(basename "$input_path")
    basename_no_ext="${filename%.*}"
    compressed_folder="${COMPRESS_OUTPUT_DIR}/${basename_no_ext}"
    
    if [ -d "$compressed_folder" ]; then
        echo ""
        log_info "正在還原: ${basename_no_ext}"
        
        # 執行 Python 解壓縮
        # 加入 -u 確保輸出即時
        python -u "$PY_DECOMPRESS" "$compressed_folder" \
            -p "$CHECKPOINT" \
            --original "$input_path"
            
        DECOMP_STATUS=$?
        
        if [ $DECOMP_STATUS -eq 0 ]; then
            ((DECOMP_COUNT++))
        else
            log_error "還原失敗: $basename_no_ext"
        fi
    else
        log_error "跳過: 找不到對應的壓縮資料夾 $compressed_folder"
    fi
done

# ==============================================================================
# 總結
# ==============================================================================
SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))

print_separator
if [ "$DECOMP_COUNT" -eq "${#PATH_ARRAY[@]}" ]; then
    log_success "全部完成! 成功處理 $DECOMP_COUNT / ${#PATH_ARRAY[@]} 個檔案"
else
    log_warning "部分完成。成功處理 $DECOMP_COUNT / ${#PATH_ARRAY[@]} 個檔案"
fi

log_info "總耗時: ${TOTAL_TIME} 秒"
log_info "結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
print_separator
