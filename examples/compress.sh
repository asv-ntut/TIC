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
export RED GREEN YELLOW BLUE NC

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

# --- 背景上傳函式 ---
upload_folder() {
    local input_path="$1"
    local base_name=$(basename "${input_path%.*}")
    local output_folder="${OUTPUT_DIR}/${base_name}"
    
    if [ -d "$output_folder" ]; then
        # 計算壓縮統計
        local original_size=$(cat "$input_path" 2>/dev/null | wc -c | tr -d ' ')
        local compressed_size=$(cat "$output_folder"/* 2>/dev/null | wc -c | tr -d ' ')
        
        if [ "$original_size" -gt 0 ] && [ "$compressed_size" -gt 0 ]; then
            local compression_ratio=$(echo "scale=1; $original_size / $compressed_size" | bc)
            echo ""
            echo -e "  ${BLUE}${base_name}${NC}"
            echo -e "    Original image: ${original_size} bytes → Compressed image: ${compressed_size} bytes (${GREEN}${compression_ratio}x${NC} compression)"
            log_info "  上傳中..."
        fi
        
        # 上傳
        scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -rq \
            "$output_folder" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" 2>/dev/null && \
            log_success "  ${base_name} 上傳完成" &
    fi
}

# --- 執行壓縮 (逐張壓縮並立即上傳) ---
log_info "正在壓縮..."
echo ""

# 將 INPUT_PATHS 轉為陣列
read -ra PATH_ARRAY <<< "$INPUT_PATHS"

TOTAL_ORIGINAL=0
TOTAL_COMPRESSED=0
UPLOAD_COUNT=0

for input_path in "${PATH_ARRAY[@]}"; do
    base_name=$(basename "${input_path%.*}")
    output_folder="${OUTPUT_DIR}/${base_name}"
    
    # 壓縮單張圖片
    python onlycompress.py "$input_path" -p "$CHECKPOINT" -o "$OUTPUT_DIR"
    COMPRESS_STATUS=$?
    
    if [ $COMPRESS_STATUS -ne 0 ]; then
        log_error "壓縮 $base_name 失敗 (exit code: $COMPRESS_STATUS)"
        continue
    fi
    
    # 累計統計
    if [ -f "$input_path" ] && [ -d "$output_folder" ]; then
        original_size=$(cat "$input_path" | wc -c | tr -d ' ')
        compressed_size=$(cat "$output_folder"/* | wc -c | tr -d ' ')
        TOTAL_ORIGINAL=$((TOTAL_ORIGINAL + original_size))
        TOTAL_COMPRESSED=$((TOTAL_COMPRESSED + compressed_size))
    fi
    
    # 立即背景上傳
    if [ "$DO_UPLOAD" = true ]; then
        upload_folder "$input_path"
        ((UPLOAD_COUNT++))
    fi
done

echo ""
log_success "壓縮完成!"

# --- 等待所有背景上傳完成 ---
if [ "$DO_UPLOAD" = true ]; then
    print_separator
    log_info "等待背景上傳完成..."
    wait  # 等待所有背景程序
    log_success "上傳完成! 共 $UPLOAD_COUNT 個資料夾"
fi

# --- 總統計 ---
print_separator
log_info "壓縮統計:"
echo ""
if [ "$TOTAL_ORIGINAL" -gt 0 ] && [ "$TOTAL_COMPRESSED" -gt 0 ]; then
    total_ratio=$(echo "scale=1; $TOTAL_ORIGINAL / $TOTAL_COMPRESSED" | bc)
    echo -e "  ${BLUE}總計${NC}"
    echo -e "    Original image: ${TOTAL_ORIGINAL} bytes → Compressed image: ${TOTAL_COMPRESSED} bytes (${GREEN}${total_ratio}x${NC} compression)"
fi
echo ""

# --- 統計 ---
SCRIPT_END=$(date +%s)
TOTAL_TIME=$((SCRIPT_END - SCRIPT_START))

print_separator
log_success "全部完成!"
log_info "結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "總耗時: ${TOTAL_TIME} 秒"
print_separator
