import torch

# 1. 指定您的權重檔案路徑
# 我們使用之前確認過表現最好的那個檔案
checkpoint_path = './pretrained/tic/3/checkpoint_best_loss.pth.tar'

# --- 完整的 try-except 結構 ---
try:
    # 2. 使用 torch.load() 嘗試讀取這個檔案
    # map_location='cpu' 參數確保即使在沒有 GPU 的電腦上也能成功讀取
    print(f"正在嘗試讀取檔案：{checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    print("檔案讀取成功！")

    # 3. 查看檔案內容的「目錄」
    # 這個檔案其實是一個 Python 的字典 (dictionary)
    # 我們可以印出它的所有「鍵」(keys)，來看看裡面打包了哪些資訊
    print("\n檔案內容的『目錄』(keys):")
    print(list(checkpoint_data.keys()))

    # 4. 查看其中一項非權重的資訊，例如 epoch
    # 這可以證明我們讀取到的資料是正確且有意義的
    if 'epoch' in checkpoint_data:
        epoch_num = checkpoint_data['epoch']
        print(f"\n這個 checkpoint 是在第 {epoch_num} 輪訓練結束後儲存的。")
    if 'loss' in checkpoint_data:
        loss_val = checkpoint_data['loss']
        print(f"儲存時的測試損失值 (loss) 為: {loss_val:.4f}")

# 這是與 try 配對的 except 區塊
except FileNotFoundError:
    print(f"\n錯誤：找不到檔案 '{checkpoint_path}'。請確認路徑和檔名是否正確。")
except Exception as e:
    print(f"\n讀取檔案時發生其他錯誤：{e}")