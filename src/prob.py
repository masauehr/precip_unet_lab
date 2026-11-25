"""確率予測の評価（numpy のみ）。

brier / bss   : ブライアスコア（確率の2乗誤差）と、気候値（学習データの発生率）に対するスキル
ece           : 信頼性の乱れ（予測確率の帯ごとに |観測頻度 − 予測確率| を件数で重み付け平均。0が理想）
reliability   : 信頼性図の表（帯ごとの 予測確率の平均・観測頻度・件数）
binned_baseline: 素の予報（ベース場）を、学習データの階級別の発生率で確率に直した基準
monotone      : しきい値が高いほど確率が下がるように補正（小さいしきい値から並べて累積最小）
best_tau      : 学習データで CSI が最大になる確率のしきい値
"""
import numpy as np


def brier(p, o):
    return float(np.mean((p - o) ** 2))


def bss(p, o, clim):
    ref = brier(np.full_like(p, clim, dtype=float), o)
    return float(1 - brier(p, o) / ref) if ref > 0 else float("nan")


def reliability(p, o, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append((p[m].mean() if m.any() else np.nan, o[m].mean() if m.any() else np.nan, int(m.sum())))
    return rows


def ece(p, o, bins=10):
    rows = reliability(p, o, bins)
    n = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[1] - r[0]) for r in rows if r[2] > 0) / n)


def binned_baseline(base_tr, o_tr, base_te, n_bins=12):
    """ベース場の対数の階級（学習データの分位点）ごとの発生率を、確率として当てはめる。"""
    lb_tr, lb_te = np.log(base_tr + 0.1), np.log(base_te + 0.1)
    edges = np.unique(np.quantile(lb_tr, np.linspace(0, 1, n_bins + 1)))
    i_tr = np.clip(np.digitize(lb_tr, edges[1:-1]), 0, len(edges) - 2)
    i_te = np.clip(np.digitize(lb_te, edges[1:-1]), 0, len(edges) - 2)
    rate = np.array([o_tr[i_tr == b].mean() if (i_tr == b).any() else o_tr.mean() for b in range(len(edges) - 1)])
    return rate[i_te]


def monotone(ps):
    """ps: しきい値の小さい順のリスト。大きいしきい値の確率が小さいしきい値を超えないようにする。"""
    return list(np.minimum.accumulate(np.stack(ps), axis=0))


def best_tau(p, o, grid=np.linspace(0.05, 0.95, 19)):
    def csi(t):
        pr = p >= t
        h, fa, ms = (pr & (o > 0)).sum(), (pr & (o == 0)).sum(), (~pr & (o > 0)).sum()
        return h / max(h + fa + ms, 1)

    return float(grid[int(np.argmax([csi(t) for t in grid]))])
