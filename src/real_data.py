"""実データ（GSM×GSMaP）の読み込みと格子合わせ。

GSMaP: 80×100（緯度0.1°×経度0.1°、北が上）。セルは 122〜132E・22〜30N を等分。
GSM  : 81×81（緯度0.1°×経度0.125°、北が上）。値は格子点にあり、点を中心とした 0.1°×0.125° のセルとみなす
       （範囲は 21.95〜30.05N・121.9375〜132.0625E で、GSMaP の領域を覆う）。
GSM を GSMaP の格子へ保存的に載せる（重なり面積で重み付けした平均。総量が保存される）。
GSMaP の欠測（NaN・-99）は、評価では GSM 側も 0 にそろえて除く（学習では損失の画素マスクにする想定）。
"""
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "data"


def overlap_matrix(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """1次元の保存的再格子化行列 (n_dst, n_src)。辺の座標は昇順。出力セルの重み和は1（領域内に収まる前提）。"""
    w = np.zeros((len(dst_edges) - 1, len(src_edges) - 1))
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i, j] = max(0.0, min(dst_edges[i + 1], src_edges[j + 1]) - max(dst_edges[i], src_edges[j]))
        w[i] /= w[i].sum()
    return w


def gsm_to_gsmap_matrices():
    """GSM(81×81) → GSMaP(80×100) の (W_y, W_x)。北が上の並びなので、緯度は符号を反転して昇順にして作る。"""
    # 緯度（北→南の並び。-lat は昇順）
    src_y = -(30.05 - 0.1 * np.arange(82))
    dst_y = -(30.0 - 0.1 * np.arange(81))
    src_x = 121.9375 + 0.125 * np.arange(82)
    dst_x = 122.0 + 0.1 * np.arange(101)
    return overlap_matrix(src_y, dst_y), overlap_matrix(src_x, dst_x)


_WY, _WX = gsm_to_gsmap_matrices()


def gsm_on_gsmap(gsm: np.ndarray) -> np.ndarray:
    """(..., 81, 81) → (..., 80, 100)。"""
    return np.einsum("ij,...jk,lk->...il", _WY, gsm, _WX).astype(np.float32)


def load_day(s: str):
    """1日分を (GSM を GSMaP 格子に載せたもの, GSMaP, 欠測マスク) で返す。形状はいずれも (24, 80, 100)。

    欠測の画素は、GSM・GSMaP ともに 0 にしてあり、マスクが True の所が欠測。
    """
    gm = np.load(D / "gsmap" / f"{s}.npy")
    bad = ~np.isfinite(gm) | (gm < 0)
    gs = gsm_on_gsmap(np.load(D / "gsm" / f"{s}.npy"))
    gm = np.where(bad, 0.0, gm).astype(np.float32)
    gs = np.where(bad, 0.0, gs).astype(np.float32)
    return gs, gm, bad
