import os
import glob
import random
import shutil
from tqdm import tqdm  # 用來顯示進度條，如果沒有請用 pip install tqdm 安裝

# ==========================================================
# ---               使用者設定區 (請修改這裡)            ---
# ==========================================================

# 1. 來源資料夾：您所有 .tif 檔案目前存放的地方
SOURCE_FOLDER = r'D:\s2_cloudfree'

# 2. 輸出資料夾：程式會在這裡建立 train, val, test 三個子資料夾
OUTPUT_FOLDER = r'D:\s2_cloudfree_split_60k'  # 建議用新名稱以區分

# 3. ✨【新功能】隨機抽樣數量
#    設定為 0 代表使用全部檔案，不進行抽樣。
SAMPLE_SIZE = 60000

# 4. 資料集劃分比例 (train, val, test)，將會應用在抽樣後的資料集上
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# 剩下的 10% 會自動分給 test

# 5. 隨機種子 (確保每次抽樣和劃分結果都一樣)
RANDOM_SEED = 42


# ==========================================================
# ---                  程式碼主體 (無需修改)             ---
# ==========================================================

def prepare_sampled_dataset_split():
    """
    掃描來源資料夾，先隨機抽樣指定數量的 .tif 檔案，
    然後再將抽樣後的檔案按比例複製到新的 train/val/test 資料夾中。
    """
    try:
        # --- 步驟 1: 定義並建立目標資料夾 ---
        print(f"來源資料夾: {SOURCE_FOLDER}")
        print(f"輸出資料夾: {OUTPUT_FOLDER}")

        train_dir = os.path.join(OUTPUT_FOLDER, 'train')
        val_dir = os.path.join(OUTPUT_FOLDER, 'val')
        test_dir = os.path.join(OUTPUT_FOLDER, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # --- 步驟 2: 從根目錄掃描所有 .tif 檔案 ---
        source_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.tif'))

        if not source_files:
            raise FileNotFoundError(f"在 '{SOURCE_FOLDER}' 中找不到任何 .tif 檔案。")

        print(f"\n在根目錄中總共找到 {len(source_files)} 個 .tif 檔案。")

        # --- 步驟 3: 隨機打亂檔案順序 ---
        random.seed(RANDOM_SEED)
        random.shuffle(source_files)

        # --- 步驟 4: ✨【新功能】從所有檔案中進行隨機抽樣 ---
        if SAMPLE_SIZE > 0 and len(source_files) > SAMPLE_SIZE:
            print(f"將從中隨機選取 {SAMPLE_SIZE} 個檔案進行下一步處理...")
            sampled_files = source_files[:SAMPLE_SIZE]
        else:
            print("檔案總數小於或等於指定的樣本數，將使用所有檔案進行處理。")
            sampled_files = source_files

        # --- 步驟 5: 計算分割數量並切分【抽樣後】的列表 ---
        total_count = len(sampled_files)
        train_count = int(total_count * TRAIN_RATIO)
        val_count = int(total_count * VAL_RATIO)

        train_files = sampled_files[:train_count]
        val_files = sampled_files[train_count: train_count + val_count]
        test_files = sampled_files[train_count + val_count:]

        print(f"規劃劃分：訓練集 {len(train_files)} 張, 驗證集 {len(val_files)} 張, 測試集 {len(test_files)} 張。")

        # --- 步驟 6: 複製檔案到對應資料夾 ---
        print("\n正在複製檔案到 'train' 資料夾...")
        for f in tqdm(train_files, desc="複製到 Train"):
            shutil.copy(f, train_dir)

        print("正在複製檔案到 'val' 資料夾...")
        for f in tqdm(val_files, desc="複製到 Validation"):
            shutil.copy(f, val_dir)

        print("正在複製檔案到 'test' 資料夾...")
        for f in tqdm(test_files, desc="複製到 Test"):
            shutil.copy(f, test_dir)

        print("\n✅ 資料集抽樣、劃分並複製完成！")

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")


if __name__ == '__main__':
    prepare_sampled_dataset_split()