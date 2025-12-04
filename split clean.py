import os
import re
import random
import shutil
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==========================================================
# ---                  使用者設定區                      ---
# ==========================================================
# 1. 來源資料夾
SOURCE_BASE_DIR = r"D:\s2_cloudfree"

# 2. 輸出資料夾
OUTPUT_BASE_DIR = r"D:\s2_cloudfree_balanced_split"

# 3. 目標 ROI 家族
ROI_FAMILIES = ['ROIs1158', 'ROIs1868', 'ROIs1970', 'ROIs2017']

# 4. 每個家族要為【訓練集】貢獻的圖片數量
TRAIN_TARGET_PER_FAMILY = 7500

# 5. 每個家族要為【驗證集】和【測試集】各貢獻的圖片數量
VAL_TARGET_PER_FAMILY = 750
TEST_TARGET_PER_FAMILY = 750

# 6. 多處理程序設定
num_processes = max(1, cpu_count() // 2)
maxtasksperchild = 1000


# ==========================================================
# ---                  工作函式 (無需修改)               ---
# ==========================================================
def copy_worker(task):
    src_path, dest_path = task
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)
    except Exception as e:
        print(f"複製檔案 {src_path} 時發生錯誤: {e}")


# ==========================================================
# ---                  主程式                            ---
# ==========================================================
def balanced_split_dataset():
    print("--- 進階平衡劃分模式 ---")

    # --- 步驟 1: 全面盤點 ---
    print("\n步驟 1/5: 全面掃描檔案並建立資料地圖...")
    all_data = defaultdict(lambda: defaultdict(list))
    pattern = re.compile(r"(ROIs\d+)_.*?_(p\d+)")
    all_files = [os.path.join(subdir, f) for subdir, _, files in os.walk(SOURCE_BASE_DIR) for f in files]

    for full_path in tqdm(all_files, desc="建立資料地圖"):
        filename = os.path.basename(full_path)
        match = pattern.search(filename)
        if match:
            roi_id, patch_id = match.groups()
            all_data[roi_id][patch_id].append(full_path)

    print("資料地圖建立完成！")

    # --- 步驟 2: 劃分 Train vs. Val/Test 候選池 ---
    print("\n步驟 2/5: 劃分訓練地點 vs. 驗證/測試候選地點...")

    train_locations = set()
    val_test_candidate_locations = defaultdict(list)

    for roi_family in ROI_FAMILIES:
        print(f"  - 處理家族: {roi_family}")
        patch_ids = list(all_data[roi_family].keys())
        random.shuffle(patch_ids)

        current_train_count = 0
        family_train_locations = set()

        for patch_id in patch_ids:
            if current_train_count < TRAIN_TARGET_PER_FAMILY:
                family_train_locations.add((roi_family, patch_id))
                current_train_count += len(all_data[roi_family][patch_id])
            else:
                val_test_candidate_locations[roi_family].append(patch_id)

        train_locations.update(family_train_locations)
        print(f"    -> 為訓練集分配了 {len(family_train_locations)} 個地點, 總計約 {current_train_count} 張圖片。")
        print(f"    -> 剩下 {len(val_test_candidate_locations[roi_family])} 個地點作為 Val/Test 候選。")

    # --- 步驟 3: 從候選池中劃分 Val vs. Test ---
    print("\n步驟 3/5: 從候選池中劃分驗證地點 vs. 測試地點...")

    val_locations = set()
    test_locations = set()

    for roi_family in ROI_FAMILIES:
        print(f"  - 處理家族: {roi_family}")
        candidate_patches = val_test_candidate_locations[roi_family]
        random.shuffle(candidate_patches)

        current_val_count = 0
        current_test_count = 0
        family_val_locs = set()
        family_test_locs = set()

        for patch_id in candidate_patches:
            if current_val_count < VAL_TARGET_PER_FAMILY:
                family_val_locs.add((roi_family, patch_id))
                current_val_count += len(all_data[roi_family][patch_id])
            elif current_test_count < TEST_TARGET_PER_FAMILY:
                family_test_locs.add((roi_family, patch_id))
                current_test_count += len(all_data[roi_family][patch_id])
            else:
                break

        val_locations.update(family_val_locs)
        test_locations.update(family_test_locs)
        print(f"    -> 為驗證集分配了 {len(family_val_locs)} 個地點, 約 {current_val_count} 張圖片。")
        print(f"    -> 為測試集分配了 {len(family_test_locs)} 個地點, 約 {current_test_count} 張圖片。")

    # --- 步驟 4: 準備複製任務列表 ---
    print("\n步驟 4/5: 準備最終複製任務列表...")

    all_tasks = []
    # 建立目標資料夾路徑
    train_dest = os.path.join(OUTPUT_BASE_DIR, 'train')
    val_dest = os.path.join(OUTPUT_BASE_DIR, 'val')
    test_dest = os.path.join(OUTPUT_BASE_DIR, 'test')

    def prepare_tasks(locations, dest_folder):
        tasks = []
        for roi_id, patch_id in locations:
            for f_path in all_data[roi_id][patch_id]:
                tasks.append((f_path, os.path.join(dest_folder, os.path.basename(f_path))))
        return tasks

    train_tasks = prepare_tasks(train_locations, train_dest)
    val_tasks = prepare_tasks(val_locations, val_dest)
    test_tasks = prepare_tasks(test_locations, test_dest)

    all_tasks = train_tasks + val_tasks + test_tasks

    print(f"最終劃分結果：")
    print(f"  - 訓練集: {len(train_tasks)} 張圖片, 來自 {len(train_locations)} 個地點。")
    print(f"  - 驗證集: {len(val_tasks)} 張圖片, 來自 {len(val_locations)} 個地點。")
    print(f"  - 測試集: {len(test_tasks)} 張圖片, 來自 {len(test_locations)} 個地點。")
    print(f"共需複製 {len(all_tasks)} 個檔案。")

    # --- 步驟 5: 並行複製 ---
    print(f"\n步驟 5/5: 開始使用 {num_processes} 個核心並行複製檔案...")

    with Pool(processes=num_processes, maxtasksperchild=maxtasksperchild) as pool:
        list(tqdm(pool.imap_unordered(copy_worker, all_tasks), total=len(all_tasks), desc="並行複製進度"))

    print("\n\n✅ 全部完成！您已成功建立一個均衡且乾淨的資料集！")
    print(f"請至以下路徑查看: {OUTPUT_BASE_DIR}")


if __name__ == '__main__':
    balanced_split_dataset()