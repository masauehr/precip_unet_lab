"""拡張した評価指標（numpy と torch のみ）。

categorical      : しきい値以上を「事象」として、的中・空振り・見逃しの数から POD・適合率・CSI・頻度バイアスを出す
fss_window       : 窓の大きさを変えた FSS（窓1＝画素どうし、大きいほど位置のずれに寛容）
cond_ratios      : 強雨の比を2通りで測る。
                   真値で条件付け: 真値がしきい値以上の画素での 予測平均/真値平均（これまでの「強雨比」）
                   予測で条件付け: 予測がしきい値以上の画素での 真値平均/予測平均（信頼性。1なら過不足なし）
spectrum_ratio   : 空間パワースペクトル（放射平均）の 予測/真値 を、波数の帯ごとに（1なら細かさが保たれている）
"""
import numpy as np
import torch
import torch.nn.functional as F


def categorical(pred, true, thr):
    p, t = pred >= thr, true >= thr
    h, fa, ms = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    return dict(
        pod=h / max(h + ms, 1),
        prec=h / max(h + fa, 1),
        csi=h / max(h + fa + ms, 1),
        fbias=(h + fa) / max(h + ms, 1),  # 予測の事象数 / 真値の事象数（1が理想）
    )


def fss_window(pred, true, thr, win):
    def frac(a):
        t = torch.tensor((a >= thr).astype(np.float32))[:, None]
        if win == 1:
            return t
        return F.avg_pool2d(t, win, stride=1, padding=win // 2, count_include_pad=False)

    pf, of = frac(pred), frac(true)
    den = (pf**2).mean() + (of**2).mean()
    return float(1 - ((pf - of) ** 2).mean() / den) if den > 0 else float("nan")


def cond_ratios(pred, true, thr):
    """(真値で条件付けた log 比, 予測で条件付けた log 比)。該当画素がなければ nan。"""
    ts, ps = true >= thr, pred >= thr
    a = np.log(max(pred[ts].mean(), 1e-9) / true[ts].mean()) if ts.any() else float("nan")
    b = np.log(max(true[ps].mean(), 1e-9) / pred[ps].mean()) if ps.any() else float("nan")
    return float(a), float(b)


def radial_spectrum(a):
    """(N, H, W) の事例ごとの2次元パワーを、放射平均して事例平均した 1次元スペクトル（波数 1〜H/2）。"""
    n, h, w = a.shape
    pw = (np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2).mean(axis=0)
    yy, xx = np.indices((h, w))
    k = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(int)
    return np.array([pw[k == i].mean() for i in range(1, h // 2 + 1)])


def spectrum_ratio(pred, true, bands=((1, 4), (5, 8), (9, 16), (17, 32))):
    """波数の帯ごとの パワー比 予測/真値（帯は 1〜4, 5〜8, 9〜16, 17〜32 周期/領域）。"""
    sp, st = radial_spectrum(pred), radial_spectrum(true)
    return [float(sp[lo - 1 : hi].sum() / st[lo - 1 : hi].sum()) for lo, hi in bands]
