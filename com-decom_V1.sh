#!/bin/bash
# ==============================================================================
# 衛星影像 壓縮/解壓縮 整合測試腳本 (PC Loopback Test)
# 功能：執行完整的 End-to-End 流程：原始影像 -> 壓縮 -> 比對統計 -> 解壓縮還原
# 適用環境：Local PC / Server
# ==============================================================================

# ==============================================================================
# 【設定區】請在這裡修改變數
# ==============================================================================

# 1. 輸入影像路徑 (原始影像，可用空格分隔多個)
# 注意：為了讓解壓縮能正確找到原始檔比對，建議使用相對路徑或絕對路徑
INPUT_PATHS="taiwan/hualien_RGB_Normalized_tile_r0_c0.tif taiwan/Taitung_RGB_Normalized_tile_r1_c0.tif"

# 2. 模型 Checkpoint 路徑
CHECKPOINT="../1130stcheckpoint_best_loss.pth"

# 3. 輸出設定
COMPRESS_OUTPUT_DIR="output_satellite"  # 壓縮後的 bitstream 存放資料夾
DECOMPRESS_OUTPUT_DIR="recon_satellite" # 解壓縮後的重建影像存放資料夾

# 4. Python 腳本名稱 (若檔名不同請在此修改)
PY_COMPRESS="onlycompress.py"
PY_DECOMPRESS="onlydecompress.py"

# ==============================================================================
# 以下為腳本邏輯
# ==============================================================================

# --- 顏色定義 (視覺化輸出) ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Log 輔助函式 ---
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_separator() { echo "============================================================"; }

# --- 環境檢查與切換 ---
# 假設腳本放在 examples 資料夾外層或與其相關的位置，這裡依照您原本的路徑邏輯設定
WORK_DIR=~/TIC/examples
VENV_ACTIVATE=~/TIC/.venv/bin/activate

# 嘗試切換目錄
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR" || { log_error "無法進入 $WORK_DIR"; exit 1; }
else
    log_warning "找不到 $WORK_DIR，假設當前目錄即為執行目錄..."
fi

# 嘗試啟動虛擬環境
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    log_warning "找不到虛擬環境 $VENV_ACTIVATE，將使用系統 Python..."
fi

# --- 開始處理 ---
echo ""
echo -e "${CYAN}"
cat << 'EOF'
      /--|  |  |--\
 |-------| (O) |-------|       AI 影像壓縮/解壓縮
 |-------| (O) |-------|         Loopback Test
 |-------| (O) |-------|     ━━━━━━━━━━━━━━━━━━━━
      \--|__|__|--/     
         |     |        
EOF
echo -e "${NC}"
print_separator
log_info "開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Checkpoint: $CHECKPOINT"
log_info "壓縮輸出: $COMPRESS_OUTPUT_DIR"
log_info "還原輸出: $DECOMPRESS_OUTPUT_DIR"
print_separator

SCRIPT_START=$(date +%s)

# ==============================================================================
# 階段一：壓縮 (Compression)
# ==============================================================================
log_step "階段 1/2: 執行壓縮..."

# 建立輸出目錄
mkdir -p "$COMPRESS_OUTPUT_DIR"

# 將 INPUT_PATHS 轉為陣列
read -ra PATH_ARRAY <<< "$INPUT_PATHS"

# 執行 Python 壓縮腳本
python "$PY_COMPRESS" "${PATH_ARRAY[@]}" -p "$CHECKPOINT" -o "$COMPRESS_OUTPUT_DIR"
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

TOTAL_ORIGINAL=0
TOTAL_COMPRESSED=0

for input_path in "${PATH_ARRAY[@]}"; do
    # 取得檔名 (不含路徑) 與去掉副檔名的名稱
    filename=$(basename "$input_path")
    basename_no_ext="${filename%.*}"
    
    # 壓縮後的資料夾路徑
    compressed_folder="${COMPRESS_OUTPUT_DIR}/${basename_no_ext}"
    
    if [ -f "$input_path" ] && [ -d "$compressed_folder" ]; then
        # 計算大小
        original_size=$(cat "$input_path" | wc -c)
        compressed_size=$(cat "$compressed_folder"/* | wc -c)
        
        # 計算比例
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
        
        TOTAL_ORIGINAL=$((TOTAL_ORIGINAL + original_size))
        TOTAL_COMPRESSED=$((TOTAL_COMPRESSED + compressed_size))
    else
        log_warning "無法讀取原始檔或壓縮檔: $input_path"
    fi
done

print_separator

# ==============================================================================
# 階段二：解壓縮 (Decompression)
# ==============================================================================
log_step "階段 2/2: 執行解壓縮與還原..."

# 建立還原目錄 (如果 Python 腳本不支援自動建立，這裡先建好)
mkdir -p "$DECOMPRESS_OUTPUT_DIR"

DECOMP_COUNT=0

for input_path in "${PATH_ARRAY[@]}"; do
    # 邏輯推導：
    # 1. 原始檔: taiwan/img.tif
    # 2. 壓縮檔資料夾: output_satellite/img (由 onlycompress.py 產生)
    
    filename=$(basename "$input_path")
    basename_no_ext="${filename%.*}"
    compressed_folder="${COMPRESS_OUTPUT_DIR}/${basename_no_ext}"
    
    if [ -d "$compressed_folder" ]; then
        echo ""
        log_info "正在還原: ${basename_no_ext}"
        
        # 組建解壓縮指令
        # 參數說明：
        #   第一個參數: 壓縮檔資料夾
        #   -p: 模型權重
        #   --original: 原始影像路徑 (用於計算 PSNR/SSIM)
        #   (選用) -o: 若您的 onlydecompress.py 支援指定輸出資料夾，可加上 -o "$DECOMPRESS_OUTPUT_DIR"
        
        python "$PY_DECOMPRESS" "$compressed_folder" \
            -p "$CHECKPOINT" \
            --original "$input_path"
            
        DECOMP_STATUS=$?
        
        if [ $DECOMP_STATUS -eq 0 ]; then
            ((DECOMP_COUNT++))
        else
            log_error "還原失敗: $basename_no_ext"
        fi
    else
        log_error "找不到對應的壓縮資料夾: $compressed_folder，跳過還原。"
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
