#!/bin/bash
# ==============================================================================
# 衛星影像批次壓縮腳本
# 功能：壓縮指定資料夾/檔案，並上傳到遠端伺服器
# ==============================================================================

# ==============================================================================
# 【設定區】請在這裡修改變數
# ==============================================================================

# 要壓縮的檔案或資料夾 (可用空格分隔多個)
# 範例: 
#   INPUT_PATHS="image.tif"                    # 單一檔案
#   INPUT_PATHS="4096_3072/"                   # 整個資料夾
#   INPUT_PATHS="img1.tif img2.tif img3.png"   # 多個檔案
INPUT_PATHS="4096_3072/hualien_RGB_Normalized_tile_r0_c0.tif 4096_3072/Taitung_RGB_Normalized_tile_r1_c0.tif"

# 模型 checkpoint 路徑
CHECKPOINT="../1130stcheckpoint_best_loss.pth"

# 輸出資料夾
OUTPUT_DIR="output_satellite"

# 是否上傳到遠端 (true/false)
DO_UPLOAD=true

# 遠端伺服器設定
REMOTE_USER="al617"
REMOTE_HOST="192.168.0.221"
REMOTE_PATH="~/TIC/TIC/examples/output_satellite/"

# ==============================================================================
# 以下為腳本邏輯，通常不需修改
# ==============================================================================

# --- 顏色定義 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_separator() { echo "============================================================"; }

# --- 環境設定 ---
cd ~/TIC/ || { log_error "找不到 ~/TIC/ 目錄"; exit 1; }
source .venv/bin/activate || { log_error "無法啟動虛擬環境"; exit 1; }
cd examples || { log_error "找不到 examples 目錄"; exit 1; }

# --- 開始處理 ---
echo ""
echo -e "${BLUE}"
cat << 'EOF'
            |
         /-----\
         | [ ] |
      /--|  |  |--\                  AI 影像壓縮系統
 |-------| (O) |-------|         ━━━━━━━━━━━━━━━━━━━━━━━
 |-------| (O) |-------|      AI Satellite Image Compression
 |-------| (O) |-------|         ━━━━━━━━━━━━━━━━━━━━━━━
      \--|__|__|--/                  NTUT 臺北科技大學
         |     |        
         \-----/
            |
EOF
echo -e "${NC}"
print_separator
log_info "開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Checkpoint: $CHECKPOINT"
log_info "輸入路徑: $INPUT_PATHS"
log_info "Downlink$( [ "$DO_UPLOAD" = true ] && echo " → ${REMOTE_USER}@${REMOTE_HOST}" || echo "否" )"
print_separator

SCRIPT_START=$(date +%s)

# --- 執行壓縮 ---
log_info "正在壓縮..."
echo ""

# 將 INPUT_PATHS 轉為陣列
read -ra PATH_ARRAY <<< "$INPUT_PATHS"
python onlycompress.py "${PATH_ARRAY[@]}" -p "$CHECKPOINT" -o "$OUTPUT_DIR"
COMPRESS_STATUS=$?

echo ""

if [ $COMPRESS_STATUS -ne 0 ]; then
    log_error "壓縮過程發生錯誤 (exit code: $COMPRESS_STATUS)"
    exit 1
fi

log_success "壓縮完成!"

# --- 計算壓縮比例 ---
print_separator
log_info "壓縮統計:"
echo ""

TOTAL_ORIGINAL=0
TOTAL_COMPRESSED=0

for input_path in "${PATH_ARRAY[@]}"; do
    base_name=$(basename "${input_path%.*}")
    output_folder="${OUTPUT_DIR}/${base_name}"
    
    if [ -f "$input_path" ] && [ -d "$output_folder" ]; then
        # 原始檔案大小 (bytes) - 使用 cat | wc -c 精確計算
        original_size=$(cat "$input_path" | wc -c)
        # 壓縮後資料夾大小 (bytes) - 使用 cat | wc -c 精確計算
        compressed_size=$(cat "$output_folder"/* | wc -c)
        
        # 計算壓縮率
        if [ "$original_size" -gt 0 ] && [ "$compressed_size" -gt 0 ]; then
            compression_ratio=$(echo "scale=1; $original_size / $compressed_size" | bc)
        else
            compression_ratio="N/A"
        fi
        
        echo -e "  ${BLUE}${base_name}${NC}"
        echo -e "    Original image: ${original_size} bytes → Compressed image: ${compressed_size} bytes (${GREEN}${compression_ratio}x${NC} compression)"
        
        TOTAL_ORIGINAL=$((TOTAL_ORIGINAL + original_size))
        TOTAL_COMPRESSED=$((TOTAL_COMPRESSED + compressed_size))
    fi
done

echo ""

# --- 上傳到遠端 ---
if [ "$DO_UPLOAD" = true ]; then
    print_separator
    log_info "正在上傳到 ${REMOTE_USER}@${REMOTE_HOST}..."
    
    UPLOAD_START=$(date +%s)
    UPLOAD_COUNT=0
    
    for input_path in "${PATH_ARRAY[@]}"; do
        # 從輸入路徑取得 basename (去掉路徑和副檔名)
        base_name=$(basename "${input_path%.*}")
        output_folder="${OUTPUT_DIR}/${base_name}"
        
        if [ -d "$output_folder" ]; then
            log_info "  上傳: $output_folder"
            scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -rq \
                "$output_folder" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" 2>/dev/null
            ((UPLOAD_COUNT++))
        else
            log_warning "  找不到: $output_folder"
        fi
    done
    
    UPLOAD_END=$(date +%s)
    UPLOAD_TIME=$((UPLOAD_END - UPLOAD_START))
    
    log_success "上傳完成! 共 $UPLOAD_COUNT 個資料夾 (耗時 ${UPLOAD_TIME} 秒)"
fi

# --- 統計 ---
SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))

print_separator
log_success "全部完成!"
log_info "結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "總耗時: ${TOTAL_TIME} 秒"
print_separator
