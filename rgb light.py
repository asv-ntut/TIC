import os
import re
import random
import shutil
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count  # --- 新增/修改 ---

# --- 設定 ---
# 1. 您原始資料的根目錄
source_dir = r"D:\s2_cloudfree"

# 2. 您要存放新資料集的目標目錄 (程式會自動建立)
dest_dir = r"D:\s2_cloudfree_custom_split"

# 3. 指定用於訓練的 ROI 家族
train_roi_family = "ROIs1158"

# 4. 指定用於驗證和測試的 ROI 家族
val_test_roi_family = "ROIs2017"

# 5. 指定驗證集和測試集的目標圖片數量
val_target_count = 2900
test_target_count = 2900

# --- 新增/修改：多處理程序設定 ---
# 使用的核心數，預設為您電腦 CPU 核心數的一半，避免電腦過於卡頓
num_processes = max(1, cpu_count() // 2)
# 每個子程序處理多少個任務後就重啟，用來防止記憶體爆掉
maxtasksperchild = 1000


# --- 新增/修改結束 ---


# --- 新增/修改：定義複製任務的「工作函式」 ---
def copy_worker(task):
    """
    這個函式將由每個獨立的子處理程序執行。
    它接收一個包含 (來源路徑, 目標路徑) 的元組。
    """
    src_path, dest_path = task
    try:
        # 確保目標資料夾存在
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)
    except Exception as e:
        # 如果複製出錯，印出錯誤訊息
        print(f"複製檔案 {src_path} 時發生錯誤: {e}")


# --- 新增/修改結束 ---


def custom_split_dataset():
    """
    根據指定規則劃分資料集
    """
    print("步驟 1/4: 開始掃描與分類所有檔案...")

    train_files = []
    rois2017_patches = defaultdict(list)
    pattern = re.compile(r"(ROIs\d+)_.*?_(p\d+)")

    all_files = [os.path.join(subdir, f) for subdir, _, files in os.walk(source_dir) for f in files]

    for full_path in tqdm(all_files, desc="掃描檔案"):
        filename = os.path.basename(full_path)
        match = pattern.search(filename)

        if match:
            roi_id = match.group(1)
            patch_id = match.group(2)

            if roi_id == train_roi_family:
                train_files.append(full_path)
            elif roi_id == val_test_roi_family:
                rois2017_patches[patch_id].append(full_path)

    print(f"\n掃描完成：")
    print(f"  - 找到 {len(train_files)} 個訓練集檔案 (來自 {train_roi_family})")
    print(f"  - 找到 {len(rois2017_patches)} 個來自 {val_test_roi_family} 的唯一地點，將從中抽取 val/test 集")

    print("\n步驟 2/4: 隨機劃分 val/test 地點ID...")

    patch_ids = list(rois2017_patches.keys())
    random.shuffle(patch_ids)

    val_files = []
    val_locations = set()
    test_files = []
    test_locations = set()

    for patch_id in patch_ids:
        if len(val_files) < val_target_count:
            val_files.extend(rois2017_patches[patch_id])
            val_locations.add(patch_id)
        elif len(test_files) < test_target_count:
            if patch_id not in val_locations:
                test_files.extend(rois2017_patches[patch_id])
                test_locations.add(patch_id)
        else:
            # 如果 val 和 test 都滿了，就提前結束迴圈
            break

    print("劃分完成：")
    print(f"  - 驗證集 (Val):   {len(val_files)} 張圖片 (來自 {len(val_locations)} 個地點)")
    print(f"  - 測試集 (Test):  {len(test_files)} 張圖片 (來自 {len(test_locations)} 個地點)")

    print("\n步驟 3/4: 準備複製任務列表...")

    # 建立目標資料夾路徑
    train_dest_folder = os.path.join(dest_dir, 'train')
    val_dest_folder = os.path.join(dest_dir, 'val')
    test_dest_folder = os.path.join(dest_dir, 'test')
    os.makedirs(train_dest_folder, exist_ok=True)
    os.makedirs(val_dest_folder, exist_ok=True)
    os.makedirs(test_dest_folder, exist_ok=True)

    # --- 新增/修改：建立一個包含所有複製任務的總列表 ---
    all_tasks = []
    # 準備訓練集的任務
    for f in train_files:
        all_tasks.append((f, os.path.join(train_dest_folder, os.path.basename(f))))
    # 準備驗證集的任務
    for f in val_files:
        all_tasks.append((f, os.path.join(val_dest_folder, os.path.basename(f))))
    # 準備測試集的任務
    for f in test_files:
        all_tasks.append((f, os.path.join(test_dest_folder, os.path.basename(f))))

    print(f"共需複製 {len(all_tasks)} 個檔案。")

    # --- 新增/修改：使用多處理程序池來執行複製 ---
    print(f"\n步驟 4/4: 開始使用 {num_processes} 個核心並行複製檔案（maxtasksperchild={maxtasksperchild}）...")

    # 建立處理程序池
    with Pool(processes=num_processes, maxtasksperchild=maxtasksperchild) as pool:
        # 使用 pool.imap_unordered 來非同步執行任務，並用 tqdm 包裝來顯示進度
        # imap_unordered 效率較高，因為它不會等待結果的順序
        list(tqdm(pool.imap_unordered(copy_worker, all_tasks), total=len(all_tasks), desc="並行複製進度"))

    print("\n全部完成！您現在可以在以下路徑找到完全按照您的規則劃分好的資料集：")
    print(dest_dir)


if __name__ == "__main__":
    custom_split_dataset()