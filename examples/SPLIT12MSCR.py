import os
import shutil
import random
from pathlib import Path
import math
import concurrent.futures
import time

# ================= 設定區 =================
SOURCE_DIR = r"D:\s2_cloudfree"

# 設定分割比例 (8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 【安全加速設定】
# 建議設定為 4 或 8。
# 設定太高 (如 32) 會導致硬碟讀寫頭過熱或系統卡死。
MAX_WORKERS = 4


# =========================================

def safe_move(file_info):
    """
    單個檔案移動的函數，給執行緒呼叫用
    file_info: (source_path, target_dir)
    """
    src, target_dir = file_info
    try:
        # 構建目標路徑
        dst = target_dir / src.name
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        # 遇到錯誤 (如檔名重複) 嘗試改名
        try:
            new_name = f"{src.stem}_dup_{int(time.time() * 1000)}{src.suffix}"
            dst = target_dir / new_name
            shutil.move(str(src), str(dst))
            return True
        except Exception as e2:
            return f"Error: {src.name} - {str(e2)}"


def flatten_and_split_dataset_parallel():
    source_path = Path(SOURCE_DIR)

    if not source_path.exists():
        print(f"❌ 錯誤：找不到路徑 {SOURCE_DIR}")
        return

    # 1. 建立目標資料夾
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        (source_path / subset).mkdir(exist_ok=True)

    print(f"🔍 正在搜尋 {SOURCE_DIR} 下的所有 .tif 檔案 (這需要一點時間)...")

    # 2. 收集所有檔案 (過濾掉已經在 train/val/test 的)
    all_tif_files = []
    for f in source_path.rglob("*.tif"):
        if f.parent.name not in subsets:
            all_tif_files.append(f)

    total_files = len(all_tif_files)
    print(f"📦 總共找到 {total_files} 個待處理檔案")

    if total_files == 0:
        print("⚠️ 沒有找到需要移動的檔案。")
        return

    # 3. 隨機打亂
    print("🎲 正在打亂數據...")
    random.seed(42)
    random.shuffle(all_tif_files)

    # 4. 計算分割
    train_count = math.floor(total_files * TRAIN_RATIO)
    val_count = math.floor(total_files * VAL_RATIO)

    train_files = all_tif_files[:train_count]
    val_files = all_tif_files[train_count: train_count + val_count]
    test_files = all_tif_files[train_count + val_count:]

    print(f"📊 分割計畫: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"🚀 啟動 {MAX_WORKERS} 個並行任務開始移動 (不會卡死電腦)...")

    # 5. 準備任務清單
    # 將 (檔案路徑, 目標資料夾) 打包成一個列表
    tasks = []
    tasks.extend([(f, source_path / 'train') for f in train_files])
    tasks.extend([(f, source_path / 'val') for f in val_files])
    tasks.extend([(f, source_path / 'test') for f in test_files])

    # 6. 多執行緒執行
    start_time = time.time()
    moved_count = 0
    total_tasks = len(tasks)

    # 使用 ThreadPoolExecutor 進行多工
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit 所有任務
        futures = [executor.submit(safe_move, task) for task in tasks]

        # 監控進度
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            moved_count += 1

            # 每搬移 2000 個檔案才印一次進度，減少螢幕輸出造成的延遲
            if moved_count % 2000 == 0:
                elapsed = time.time() - start_time
                speed = moved_count / elapsed
                print(f"   [進度 {moved_count}/{total_tasks}] - 速度: {speed:.0f} 檔/秒")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ 全部完成！耗時: {duration:.2f} 秒")
    print(f"📂 請檢查 {SOURCE_DIR} 下的 train, val, test 資料夾。")
    print("🧹 提示：原本的空資料夾現在可以手動刪除了。")


if __name__ == "__main__":
    # 這裡加一個保護，防止 Windows 下多進程出錯 (雖然這裡是多線程，但好習慣)
    flatten_and_split_dataset_parallel()