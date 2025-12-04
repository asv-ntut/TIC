# 匯入需要的函式庫
import os
from PIL import Image

# --- 設定 ---
# 1. 指定包含原始圖片的資料夾
input_folder = 'C:/Users/Matt/Desktop/inference'

# 2. 指定您要處理的完整檔案名稱
#    請確保這個檔名和您資料夾中的完全一致
filename = 'STUPID PHOTO.jpg'

# 3. 處理完後要儲存的新檔名
output_filename = 'resized_image.png'

# 4. 您想要的目標尺寸
target_size = (256, 256)

# --- 主程式邏輯 ---
# 組合出完整的輸入檔案路徑
input_path = os.path.join(input_folder, filename)

print(f"準備處理單一檔案：{input_path}")

try:
    # 開啟圖片檔案
    with Image.open(input_path) as img:
        # 將圖片縮放至目標尺寸
        resized_img = img.resize(target_size)

        # 確保圖片是 RGB 格式 (3個通道)，去除透明度(A)等資訊
        if resized_img.mode != 'RGB':
            resized_img = resized_img.convert('RGB')

        # 儲存縮放後的圖片
        resized_img.save(output_filename)

        print(f"\n處理完成！")
        print(f"原始圖片：{input_path}")
        print(f"已成功縮放並儲存為：{output_filename}")

except FileNotFoundError:
    # 如果找不到指定的檔案，會提示錯誤
    print(f"\n錯誤：在 '{input_folder}' 資料夾中找不到名為 '{filename}' 的檔案。")
    print("請確認檔案是否存在，且檔名完全正確。")
except Exception as e:
    # 如果發生其他讀取錯誤
    print(f"\n處理檔案時發生錯誤: {e}")