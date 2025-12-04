import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# --- 路徑已根據您的詳細資訊修改完成 ---
# 推薦使用「絕對路徑」，注意路徑最前面的 r 很重要，它可以防止跳脫字元問題
# 這種寫法，不論你的 .py 檔放哪裡都可以執行
filepath = r'D:\s2_cloudfree\ROIs1158_spring_1_p33.tif'
# -----------------------------------------

try:
    # 使用 with 陳述式來安全地開啟檔案
    with rasterio.open(filepath) as src:

        print("--- 檔案基本資訊 ---")
        print(f"檔案格式: {src.driver}")
        print(f"波段數量 (Bands): {src.count}")
        # ... (後續程式碼與之前相同，此處省略以保持簡潔) ...
        print(f"影像寬度 (Width): {src.width} 像素")
        print(f"影像高度 (Height): {src.height} 像素")
        print(f"資料類型 (Data Type): {src.dtypes[0]}")
        print(f"座標參考系統 (CRS): {src.crs}")
        print(f"是否有色彩映射表 (Colormap): {'是' if src.colorinterp else '否'}")

        if src.width == 0 or src.height == 0:
            print("\n警告：影像的寬度或高度為 0，檔案內容可能為空！")
        else:
            data = src.read(1)
            print(f"\n成功讀取第一個波段，資料維度: {data.shape}")
            print("正在嘗試繪製影像...")
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            show(src, ax=ax)
            ax.set_title("Rasterio 視覺化結果")
            plt.show()

except rasterio.errors.RasterioIOError as e:
    print(f"\n錯誤：無法使用 Rasterio 開啟檔案 '{filepath}'。")
    print("這可能是因為：")
    print("1. 檔案路徑錯誤或檔案不存在。")
    print("2. 檔案已損毀。")
    print("3. 這不是一個 Rasterio 支援的地理空間光柵格式。")
    print(f"詳細錯誤訊息: {e}")

except Exception as e:
    print(f"\n發生未預期的錯誤: {e}")

