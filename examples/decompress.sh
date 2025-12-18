#!/bin/bash
# ==============================================================================
# 衛星影像自動解壓縮腳本
# 功能：監控資料夾變更，當偵測到新的 .bin 檔案時自動解壓縮
# ==============================================================================

# ==============================================================================
# 【設定區】
# ==============================================================================

# 監控的資料夾
WATCH_DIR="output_satellite"

# 模型 checkpoint 路徑
CHECKPOINT="../1130stcheckpoint_best_loss.pth"

# 原始圖片目錄 (用於計算 PSNR，若不需要可留空)
ORIGINAL_DIR="taiwan"

# 掃描間隔 (秒)
SCAN_INTERVAL=5

# ==============================================================================
# 以下為腳本邏輯
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# 切換到腳本所在目錄
cd "$(dirname "$0")" || exit 1

# 建立監控目錄
mkdir -p "$WATCH_DIR"

# ==============================================================================
# 信號處理 (允許 Ctrl+C 中斷)
# ==============================================================================
cleanup() {
    echo ""
    log_warning "收到中斷信號，正在退出..."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ==============================================================================
# 記錄已處理的資料夾
# ==============================================================================
PROCESSED_FILE="/tmp/decompress_processed_$$"
touch "$PROCESSED_FILE"

is_processed() {
    grep -qxF "$1" "$PROCESSED_FILE" 2>/dev/null
}

mark_processed() {
    echo "$1" >> "$PROCESSED_FILE"
}

# ==============================================================================
# 取得資料夾的 .bin 檔案數量
# ==============================================================================
get_bin_count() {
    local folder="$1"
    find "$folder" -maxdepth 1 -name "*.bin" 2>/dev/null | wc -l | tr -d ' '
}

# ==============================================================================
# 解壓縮單一資料夾
# ==============================================================================
decompress_folder() {
    local folder_path="$1"
    local folder_name=$(basename "$folder_path")
    
    log_info "解壓縮: $folder_name"
    
    # 嘗試找到對應的原始圖片
    original_file=""
    if [ -n "$ORIGINAL_DIR" ]; then
        for ext in tif tiff png jpg; do
            candidate="${ORIGINAL_DIR}/${folder_name}.${ext}"
            if [ -f "$candidate" ]; then
                original_file="$candidate"
                break
            fi
        done
    fi
    
    # 執行解壓縮
    if [ -n "$original_file" ]; then
        log_info "  原始檔案: $original_file"
        python3 onlydecompress.py "$folder_path" -p "$CHECKPOINT" --original "$original_file"
    else
        log_info "  (無原始檔案，跳過 PSNR 計算)"
        python3 onlydecompress.py "$folder_path" -p "$CHECKPOINT"
    fi
    
    return $?
}

# ==============================================================================
# 主程式
# ==============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}         衛星影像自動解壓縮系統${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log_info "監控目錄: ${WATCH_DIR}"
log_info "掃描間隔: ${SCAN_INTERVAL} 秒"
log_info "按 Ctrl+C 可中斷"
echo ""

# 記錄每個資料夾的 .bin 檔案數量 (用於偵測穩定)
declare -A FOLDER_COUNTS

while true; do
    CHANGED=false
    
    # 掃描所有子資料夾
    for folder in "$WATCH_DIR"/*/; do
        [ -d "$folder" ] || continue
        
        folder_name=$(basename "$folder")
        
        # 已處理過的跳過
        if is_processed "$folder_name"; then
            continue
        fi
        
        # 取得目前 .bin 檔案數量
        current_count=$(get_bin_count "$folder")
        
        # 如果沒有 .bin 檔案，跳過
        [ "$current_count" -eq 0 ] && continue
        
        # 取得上次記錄的數量
        last_count="${FOLDER_COUNTS[$folder_name]:-0}"
        
        if [ "$current_count" != "$last_count" ]; then
            # 數量有變化，可能還在傳輸中
            FOLDER_COUNTS[$folder_name]="$current_count"
            echo -ne "\r${BLUE}[偵測]${NC} $folder_name: $current_count 個封包 (傳輸中...)    "
            CHANGED=true
        else
            # 數量穩定，可以開始解壓縮
            if [ "$current_count" -gt 0 ]; then
                echo ""
                log_success "偵測到完整資料: $folder_name ($current_count 個封包)"
                echo ""
                
                decompress_folder "$folder"
                
                if [ $? -eq 0 ]; then
                    mark_processed "$folder_name"
                    log_success "$folder_name 解壓縮完成!"
                else
                    log_error "$folder_name 解壓縮失敗"
                fi
                echo ""
                CHANGED=true
            fi
        fi
    done
    
    # 如果沒有變化，顯示等待訊息
    if [ "$CHANGED" = false ]; then
        echo -ne "\r${BLUE}[等待]${NC} 掃描中... ($(date '+%H:%M:%S'))    "
    fi
    
    sleep "$SCAN_INTERVAL"
done

# 清理
rm -f "$PROCESSED_FILE"
