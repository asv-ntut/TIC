import shutil
from pathlib import Path

def copy_top_n_tifs(src_dir, dest_dir, limit=100):
    # 將路徑字串轉換為 Path 物件
    source_path = Path(src_dir)
    target_path = Path(dest_dir)

    # 檢查來源資料夾是否存在
    if not source_path.exists():
        print(f"錯誤：找不到來源資料夾 '{src_dir}'")
        return

    # 如果目標資料夾不存在，則自動建立 (parents=True 允許建立多層結構)
    target_path.mkdir(parents=True, exist_ok=True)

    # 找出所有 .tif 檔案（不分大小寫可用 .glob('*.[tT][iI][fF]')）
    # 並依照檔名排序，確保「前 100 張」是固定的
    tif_files = sorted(source_path.glob('*.tif'))

    # 選取前 N 張 (如果少於 N 張，則選取全部)
    files_to_copy = tif_files[:limit]

    total_files = len(files_to_copy)
    if total_files == 0:
        print("來源資料夾中沒有找到 .tif 檔案。")
        return

    print(f"開始複製 {total_files} 個檔案到 {dest_dir} ...")

    # 執行複製
    success_count = 0
    for file_path in files_to_copy:
        try:
            # shutil.copy2 會連同 metadata (時間戳記等) 一起複製
            shutil.copy2(file_path, target_path / file_path.name)
            success_count += 1
        except Exception as e:
            print(f"複製 {file_path.name} 時發生錯誤: {e}")

    print(f"完成！成功複製 {success_count}/{total_files} 個檔案。")

# ==========================================
# 參數設定區 (請修改這裡的路徑)
# ==========================================
SOURCE_DIRECTORY = "/home/asvserver/TIC/s2_combined/test"  # 你的原始資料夾路徑
TARGET_DIRECTORY = "/home/asvserver/TIC/s2_combined/test_16bit_192"  # 新的資料夾路徑

if __name__ == "__main__":
    copy_top_n_tifs(SOURCE_DIRECTORY, TARGET_DIRECTORY, limit=192)