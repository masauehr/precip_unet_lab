"""再格子化（保存的・双線形）。numpy のみで動く。

格子は「領域 [0, 1] を等分したセル」とみなす。縦横それぞれ1次元の行列を作り、
2次元場には W_y @ a @ W_x.T として適用する（分離可能）。
"""
import numpy as np


def conservative_matrix(n_src: int, n_dst: int) -> np.ndarray:
    """保存的再格子化の1次元行列 (n_dst, n_src)。

    出力セルの値 = 入力セルとの重なり長で重み付けした平均。
    各行の和は1。領域全体の平均（総量）が保存される。
    """
    es = np.linspace(0.0, 1.0, n_src + 1)
    ed = np.linspace(0.0, 1.0, n_dst + 1)
    w = np.zeros((n_dst, n_src))
    for i in range(n_dst):
        for j in range(n_src):
            w[i, j] = max(0.0, min(ed[i + 1], es[j + 1]) - max(ed[i], es[j]))
        w[i] /= w[i].sum()
    return w


def bilinear_matrix(n_src: int, n_dst: int) -> np.ndarray:
    """双線形補間の1次元行列 (n_dst, n_src)。セル中心で補間し、端は端の値を延長する。

    注意: 降水のような「量」には使わない（総量が保存されない）。地形などの連続量向け。
    """
    xs = (np.arange(n_src) + 0.5) / n_src
    xd = (np.arange(n_dst) + 0.5) / n_dst
    w = np.zeros((n_dst, n_src))
    eye = np.eye(n_src)
    for j in range(n_src):
        w[:, j] = np.interp(xd, xs, eye[j])
    return w


def _apply(a: np.ndarray, wy: np.ndarray, wx: np.ndarray) -> np.ndarray:
    """最後の2軸（緯度・経度方向）に行列を適用する。"""
    return np.einsum("ij,...jk,lk->...il", wy, a, wx)


def regrid_conservative(a: np.ndarray, shape: tuple) -> np.ndarray:
    """保存的再格子化。a の最後の2軸を shape=(ny, nx) に変換する。"""
    ny, nx = a.shape[-2:]
    return _apply(a, conservative_matrix(ny, shape[0]), conservative_matrix(nx, shape[1]))


def regrid_bilinear(a: np.ndarray, shape: tuple) -> np.ndarray:
    """双線形補間。a の最後の2軸を shape=(ny, nx) に変換する。"""
    ny, nx = a.shape[-2:]
    return _apply(a, bilinear_matrix(ny, shape[0]), bilinear_matrix(nx, shape[1]))
