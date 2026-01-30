import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. 讀取 CSV 檔案
    csv_file = "benchmark_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {csv_file}，請確認檔案在同一目錄下。")
        return

    # 2. 分離 JPEG 與 JPEG2000 的數據
    jpeg_df = df[df['Method'] == 'JPEG']
    jp2_df = df[df['Method'] == 'JPEG2000']

    # 3. 設定畫布 (1列, 2欄)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TIC Project Baseline RD Curves', fontsize=16, fontweight='bold')

    # ==========================================
    # 圖 1: BPP vs MS-SSIM
    # ==========================================
    ax1 = axes[0]
    ax1.plot(jpeg_df['Avg_BPP'], jpeg_df['Avg_MS-SSIM'], '-o', label='JPEG', linewidth=2, markersize=8)
    ax1.plot(jp2_df['Avg_BPP'], jp2_df['Avg_MS-SSIM'], '-s', label='JPEG2000', linewidth=2, markersize=8)
    
    ax1.set_title('Rate-Distortion Curve (MS-SSIM)', fontsize=14)
    ax1.set_xlabel('Bit-rate (BPP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MS-SSIM', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=12)

    # ==========================================
    # 圖 2: BPP vs PSNR (需過濾 inf 值)
    # ==========================================
    ax2 = axes[1]
    
    # 畫 JPEG (假設 JPEG 沒有 inf)
    ax2.plot(jpeg_df['Avg_BPP'], jpeg_df['Avg_PSNR'], '-o', label='JPEG', linewidth=2, markersize=8)

    # 過濾 JPEG2000 中的 inf 值
    jp2_valid_psnr = jp2_df[~np.isinf(jp2_df['Avg_PSNR'])]
    if not jp2_valid_psnr.empty:
        ax2.plot(jp2_valid_psnr['Avg_BPP'], jp2_valid_psnr['Avg_PSNR'], '-s', label='JPEG2000', linewidth=2, markersize=8)

    ax2.set_title('Rate-Distortion Curve (PSNR)', fontsize=14)
    ax2.set_xlabel('Bit-rate (BPP)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right', fontsize=12)

    # 標註提示：說明 JPEG2000 是 Lossless 的狀況
    ax2.text(0.5, 0.1, 'Note: JPEG2000 PSNR = $\infty$ (Lossless)\nNot shown on plot.', 
             transform=ax2.transAxes, fontsize=12, color='red',
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

    # 自動調整版面並顯示
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # 留出空間給大標題
    
    # 存檔 (論文用向量圖與預覽圖)
    plt.savefig('_rd_curves.png', dpi=300)
    print("圖表已存檔為: _rd_curves.png")
    
    plt.show()

if __name__ == "__main__":
    main()