cd ..
python onlydecompress.py "output_satellite/hualien_RGB_Normalized_tile_r0_c0" -p "./1130stcheckpoint_best_loss.pth" --original "taiwan/hualien_RGB_Normalized_tile_r0_c0.tif"
python onlydecompress.py "output_satellite/Taitung_RGB_Normalized_tile_r1_c0" -p "./1130stcheckpoint_best_loss.pth" --original "taiwan/Taitung_RGB_Normalized_tile_r1_c0.tif"