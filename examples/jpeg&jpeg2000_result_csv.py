import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. 讀取 CSV 檔案
    csv_file = "benchmark_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {csv_file}")
        return

    # 2. 數據格式化：處理小數點與 inf
    # 將 BPP 和 MS-SSIM 四捨五入到小數點後 4 位
    df['Avg_BPP'] = df['Avg_BPP'].apply(lambda x: f"{x:.4f}")
    df['Avg_MS-SSIM'] = df['Avg_MS-SSIM'].apply(lambda x: f"{x:.4f}")
    
    # 將 PSNR 中的 inf 替換為 ∞ 符號，其餘保留 2 位小數
    df['Avg_PSNR'] = df['Avg_PSNR'].apply(
        lambda x: "∞ (Lossless)" if np.isinf(x) else f"{x:.2f}"
    )

    # 重新命名欄位，讓表格標頭更易讀
    display_df = df.rename(columns={
        'Method': 'Codec',
        'Parameter': 'Setting',
        'Avg_BPP': 'BPP',
        'Avg_PSNR': 'PSNR (dB)',
        'Avg_MS-SSIM': 'MS-SSIM'
    })

    # --- 輸出 1：終端機文字表格 ---
    print("\n" + "="*50)
    print(" 壓縮演算法效能對比表 (Baseline)")
    print("="*50)
    print(display_df.to_markdown(index=False)) # 需要安裝 tabulate 套件
    print("="*50 + "\n")

    # --- 輸出 2：繪製並儲存表格圖片 ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # 隱藏座標軸
    
    # 建立表格
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center'
    )

    # 美化表格樣式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8) # 調整格子大小 (寬度, 高度)

    # 將標題列設為粗體並加上底色
    for i in range(len(display_df.columns)):
        cell = table[0, i]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4') # 使用標準的學術藍色

    # 標題
    plt.title("Baseline Rate-Distortion Benchmark", fontweight='bold', fontsize=14, pad=10)

    # 存檔
    plt.tight_layout()
    plt.savefig('_benchmark_table.png', dpi=300, bbox_inches='tight')
    print("✅ 成功！表格已儲存為圖檔：benchmark_table.png")

if __name__ == "__main__":
    main()