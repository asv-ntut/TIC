/* Copyright (c) 2021-2022, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
   * although it's only run once per model after training. See TF/compression
   * implementation for an optimized version. */

  for (float p : pmf) {
    if (p < 0 || !std::isfinite(p)) {
      throw std::domain_error(
          std::string("Invalid `pmf`, non-finite or negative element found: ") +
          std::to_string(p));
    }
  }

  std::vector<uint32_t> cdf(pmf.size() + 1);
  cdf[0] = 0; /* freq 0    程式碼首先會檢查 pmf 裡沒有負數或無效值。然後建立一個大小為 3 + 1 = 4 的 cdf 向量，並將第一個元素設為 0。
              cdf 向量目前是：{0, ?, ?, ?}*/

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                 [=](float p) { return std::round(p * (1 << precision)); });
/*轉換 PMF 為整數頻率 (Quantization)：

1 << precision 就是 2
4
 =16。

這個 transform 操作會遍歷 pmf 中的每個元素 p，計算 p * 16 並四捨五入。

pmf[0] = 0.1 -> round(0.1 * 16) = round(1.6) = 2

pmf[1] = 0.7 -> round(0.7 * 16) = round(11.2) = 11

pmf[2] = 0.2 -> round(0.2 * 16) = round(3.2) = 3

這些結果被填入 cdf 從第 1 個元素開始的位置。 */
/*此時 cdf 的狀態: {0, 2, 11, 3}

注意：這時候 cdf 向量暫時存的是頻率 (Frequency)，而不是累積值 (Cumulative)。2, 11, 3 分別代表三個符號在 16 的總範圍中所佔的份額。*/

  const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);
  if (total == 0) {
    throw std::domain_error("Invalid `pmf`: at least one element must have a "
                            "non-zero probability.");
  }

  std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                 [precision, total](uint32_t p) {
                   return ((static_cast<uint64_t>(1 << precision) * p) / total);
                 });

  std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
  cdf.back() = 1 << precision;
  /*std::partial_sum: 這個函式會就地計算累積和。

cdf[0] -> 0 (不變)

cdf[1] -> cdf[0] + cdf[1] = 0 + 2 = 2

cdf[2] -> cdf[1] + cdf[2] = 2 + 11 = 13

cdf[3] -> cdf[2] + cdf[3] = 13 + 3 = 16

cdf.back() = 1 << precision: 強制設定最後一個元素為 16，確保 CDF 的範圍是完整的 [0, 16]，消除任何潛在的浮點數誤差。

此時 cdf 的狀態: {0, 2, 13, 16} */

  for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
    if (cdf[i] == cdf[i + 1]) { //確保 沒有空元素
      /* Try to steal frequency from low-frequency symbols */
      uint32_t best_freq = ~0u;/*初始化為最大整數*/
      int best_steal = -1;/*初始化為無效索引*/
      for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
        uint32_t freq = cdf[j + 1] - cdf[j];
        if (freq > 1 && freq < best_freq) {
          best_freq = freq;
          best_steal = j;/*最好的金主*/
        }
      }
      /*for (int j = 0; ...): 這個迴圈的任務是掃描所有符號，找出一個最適合被「偷」的對象。

j: 代表候選金主的索引。

freq: cdf[j+1] - cdf[j]，計算第 j 個符號的當前頻率。

if (freq > 1 && freq < best_freq): 這是挑選「最佳金主」的條件。

freq > 1: 必須的！ 我們不能從一個頻率只有 1 的符號偷，不然它自己就變 0 了。

freq < best_freq: 策略性的。如果有好幾個符號的頻率都大於 1，我們傾向於從那個頻率比較小的偷（但仍要大於1），這樣對整個機率分佈的影響最小。

best_freq 和 best_steal: 用來記錄目前為止找到的最佳人選的「頻率」和「索引」。 */

      assert(best_steal != -1);

      if (best_steal < i) { //金主在窮人前面
        for (int j = best_steal + 1; j <= i; ++j) {
          cdf[j]--;
        }
      } else {
        assert(best_steal > i);//金主在窮人後面
        for (int j = i + 1; j <= best_steal; ++j) {
          cdf[j]++;
        }
      }
    }
  }

  assert(cdf[0] == 0);
  assert(cdf.back() == (1 << precision));// 確保 終點是2的n次方
  for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
    assert(cdf[i + 1] > cdf[i]);// 確保 後面得 都比前面的大
  }

  return cdf;
}

PYBIND11_MODULE(_CXX, m) {
  m.attr("__name__") = "compressai._CXX";

  m.doc() = "C++ utils";

  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf,
        "Return quantized CDF for a given PMF");
}
