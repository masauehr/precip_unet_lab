"""GSM(FT=k) と GSMaP の時刻整合を、ラグ相関で確かめる。

両方を 1° の箱（8×10 個）の平均に揃え、GSM の時刻 t（日の通し時間）と GSMaP の t+lag を比べる。
使い方: .venv/bin/python src/check_align.py
"""
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "data"


import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")


def corr(a, b):
    """両方が有効な要素だけで相関を取る。"""
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def box(a, ny, nx):
    """(T, H, W) を ny×nx 個の箱の平均にする（欠測 NaN は除く。箱が全部欠測なら NaN）。"""
    t, h, w = a.shape
    by, bx = h // ny, w // nx
    return np.nanmean(a[:, :by * ny, :bx * nx].reshape(t, ny, by, nx, bx), axis=(2, 4))


def load_gsmap(d):
    """GSMaP の1日分。-99（データなしの符号）は NaN にする（2025-10-10 は全面 -99）。"""
    a = np.load(D / "gsmap" / f"{d}.npy")
    a[a < 0] = np.nan
    return a


def load_pairs():
    days = sorted(p.stem for p in (D / "gsmap").glob("2*.npy") if (D / "gsm" / p.name).exists())
    gm = np.stack([box(load_gsmap(d), 8, 10) for d in days])   # (日,24,8,10)
    gs = np.stack([box(np.load(D / "gsm" / f"{d}.npy")[:, :80, :80], 8, 10) for d in days])
    return days, gm, gs


if __name__ == "__main__":
    days, gm, gs = load_pairs()
    n = len(days)
    # 日をつないだ通し時間（日×24）にする。GSMaP は同じ日の hh 時＝その時の開始
    gm_t = gm.reshape(n * 24, 8, 10)
    gs_t = gs.reshape(n * 24, 8, 10)
    print(f"日数 {n}（{days[0]}〜{days[-1]}）  平均 GSMaP {np.nanmean(gm):.3f}  GSM {gs.mean():.3f} mm/h")
    print("ラグ: GSM の時刻 t（FT=t+1）と GSMaP の時刻 t+lag の相関")
    print(f"{'lag':>4}{'領域平均の相関':>16}{'箱ごとの相関':>14}")
    for lag in range(-4, 5):
        if lag >= 0:
            a, b = gs_t[:len(gs_t) - lag], gm_t[lag:]
        else:
            a, b = gs_t[-lag:], gm_t[:len(gm_t) + lag]
        r_mean = corr(a.mean(axis=(1, 2)), b.mean(axis=(1, 2)))
        r_box = corr(a.ravel(), b.ravel())
        print(f"{lag:>4}{r_mean:>16.3f}{r_box:>14.3f}")
    print("\n予報時間 FT 帯別（lag=0）の箱ごとの相関と平均降水")
    for lo, hi in [(0, 6), (6, 12), (12, 18), (18, 24)]:
        a, b = gs[:, lo:hi], gm[:, lo:hi]
        print(f"FT {lo + 1:>2}〜{hi:>2}h  相関 {corr(a.ravel(), b.ravel()):.3f}  GSM {a.mean():.3f}  GSMaP {np.nanmean(b):.3f}")
