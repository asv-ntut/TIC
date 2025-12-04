import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==========================================================
# ---               使用者設定區 (請修改這裡)            ---
# ==========================================================
# 請指定一張您處理完的【3通道 RGB TIF】檔案的完整路徑
IMAGE_PATH = r'D:\s2_cloudfree_rgb_60k\train\ROIs1158_spring_100_p29.tif'  # 我填上了您最新的範例


# ==========================================================

def run_ultimate_display_test():
    """
    執行一個兩步驟的測試，來判斷問題來源。
    """

    # --- 測試一：顯示一張電腦自己畫的圖 ---
    print("--- 測試一：顯示電腦合成圖 ---")
    print("這個測試用來確認您的 Matplotlib 顯示環境是否正常。")

    try:
        # 建立一個 256x256x3 的空白畫布
        synthetic_image = np.zeros((256, 256, 3), dtype=np.float32)

        # 左半邊填滿紅色 (R=1, G=0, B=0)
        synthetic_image[:, :128, 0] = 1.0

        # 右半邊填滿綠色 (R=0, G=1, B=0)
        synthetic_image[:, 128:, 1] = 1.0

        plt.figure(figsize=(6, 6))
        plt.imshow(synthetic_image)
        plt.title("測試一：電腦合成圖\n(如果正常，應為左紅右綠)")
        plt.show()
        print(" -> 測試一視窗已顯示。")

    except Exception as e:
        print(f"❌ 測試一發生錯誤: {e}")
        return  # 如果這裡出錯，後續就不用測了

    # --- 測試二：用最基礎的方法顯示您的 TIF 檔 ---
    print("\n--- 測試二：用最基礎的方法顯示您的 TIF 檔 ---")
    print("這個測試用來確認您的 TIF 資料在經過最簡單的亮度拉伸後，是否能被看見。")

    try:
        with rasterio.open(IMAGE_PATH) as src:
            # 讀取 uint16 原始數據，並轉成 float32
            img_data = src.read().astype(np.float32)

        # 轉換維度 (C, H, W) -> (H, W, C)
        display_data = np.transpose(img_data, (1, 2, 0))

        # 使用最基礎、最簡單的「最小-最大值」方法來正規化
        min_val = np.min(display_data)
        max_val = np.max(display_data)

        if max_val > min_val:
            normalized_data = (display_data - min_val) / (max_val - min_val)
        else:
            normalized_data = display_data - min_val

        plt.figure(figsize=(6, 6))
        plt.imshow(normalized_data)
        plt.title(f"測試二：您的 TIF 檔案\n(基礎亮度拉伸後)")
        plt.show()
        print(" -> 測試二視窗已顯示。")

    except Exception as e:
        print(f"❌ 測試二發生錯誤: {e}")


if __name__ == '__main__':
    run_ultimate_display_test()