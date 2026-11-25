"""合成データ: 真の細かい降水場と、誤差を含む粗い予報を作る。

真の場（fine）  : ガウス状の降水セルの重ね合わせ＋乗法ノイズ。0 mm/h の領域を多く含む。
粗い予報（coarse）: 真の場の面積平均（保存的）に、系統的な過大/過小・位置ずれ・ばらつきを加える。
→ 実データ（解析雨量 × 数値予報）に置き換える前に、手法の挙動を確かめるための題材。
"""
import numpy as np

from regrid import regrid_conservative


def make_fine_field(rng: np.random.Generator, n: int = 64) -> np.ndarray:
    """真の細かい降水場 (n, n) [mm/h]。"""
    yy, xx = np.mgrid[0:n, 0:n]
    field = np.zeros((n, n))
    n_cells = rng.integers(0, 7)  # 0個なら無降水の事例
    for _ in range(n_cells):
        cy, cx = rng.uniform(0, n, size=2)
        s = rng.uniform(1.5, 6.0) * n / 64
        amp = rng.lognormal(mean=1.5, sigma=0.9)
        field += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s**2))
    field *= rng.lognormal(mean=0.0, sigma=0.3, size=(n, n))  # 乗法ノイズ
    field[field < 0.1] = 0.0
    return field


def make_coarse_forecast(
    fine: np.ndarray, factor: int, rng: np.random.Generator
) -> np.ndarray:
    """粗い予報 (n/factor, n/factor)。面積平均に、過大バイアス・位置ずれ・ばらつきを加える。"""
    n = fine.shape[-1]
    nc = n // factor
    coarse = regrid_conservative(fine, (nc, nc))
    # 系統的な過大（中心 +25%）と事例ごとのばらつき
    coarse = coarse * rng.lognormal(mean=0.22, sigma=0.3)
    # 位置ずれ: 粗い格子で 0〜1 セル
    dy, dx = rng.integers(-1, 2, size=2)
    coarse = np.roll(coarse, shift=(dy, dx), axis=(0, 1))
    # セル単位の予報誤差
    coarse = coarse * rng.lognormal(mean=0.0, sigma=0.25, size=coarse.shape)
    return coarse


def make_dataset(n_samples: int, n: int = 64, factor: int = 8, seed: int = 0):
    """(coarse, fine) の組を返す。形状は (N, nc, nc) と (N, n, n)。"""
    rng = np.random.default_rng(seed)
    fines = np.stack([make_fine_field(rng, n) for _ in range(n_samples)])
    coarses = np.stack([make_coarse_forecast(f, factor, rng) for f in fines])
    return coarses, fines
