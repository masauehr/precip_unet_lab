"""実データの基準評価: GSM（保存的に GSMaP 格子へ載せたもの）を、そのまま・総量を合わせた場合で評価する。

評価するのは 学習集合と検証集合だけ。**テスト集合は最終評価の1回まで見ない**（読み込まない）。
指標は日ごとの合計量にしてから、ブロック単位のブートストラップで 95% 区間を出す。
  対数RMSE、総量比（予測/真値）、しきい値別 POD・FAR・CSI・頻度バイアス、FSS（窓1・9）
使い方: .venv/bin/python src/real_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS
from real_data import load_day
from split import make_split

THRS = (1.0, 5.0, 10.0)
WINS = (1, 9)


def frac(a, thr, win):
    t = torch.tensor((a >= thr).astype(np.float32))[:, None]
    return t if win == 1 else F.avg_pool2d(t, win, stride=1, padding=win // 2, count_include_pad=False)


def day_stats(pred, true, bad):
    """1日分（24,80,100）の合計量。ブロック単位で足し合わせて指標にする。欠測画素は数えない。"""
    ok = ~bad
    d = (np.log(pred + EPS) - np.log(true + EPS)) ** 2
    s = {"sq": float(d[ok].sum()), "n": float(ok.sum()), "pred": float(pred[ok].sum()), "true": float(true[ok].sum())}
    for t in THRS:
        p, o = (pred >= t) & ok, (true >= t) & ok
        s[f"h{t}"], s[f"fa{t}"], s[f"ms{t}"] = float((p & o).sum()), float((p & ~o).sum()), float((~p & o).sum())
        for w in WINS:
            pf, of = frac(pred, t, w), frac(true, t, w)
            s[f"fn{t}_{w}"] = float(((pf - of) ** 2).sum())
            s[f"fd{t}_{w}"] = float((pf**2).sum() + (of**2).sum())
    return s


def metrics(S):
    """日ごとの合計量（辞書のリスト）を足して指標にする。"""
    g = lambda k: sum(x[k] for x in S)
    m = {"対数RMSE": np.sqrt(g("sq") / g("n")), "総量比": g("pred") / g("true")}
    for t in THRS:
        h, fa, ms = g(f"h{t}"), g(f"fa{t}"), g(f"ms{t}")
        m[f"POD{t:g}"] = h / max(h + ms, 1)
        m[f"FAR{t:g}"] = fa / max(h + fa, 1)
        m[f"CSI{t:g}"] = h / max(h + fa + ms, 1)
        m[f"頻度比{t:g}"] = (h + fa) / max(h + ms, 1)
        for w in WINS:
            m[f"FSS{t:g}_w{w}"] = 1 - g(f"fn{t}_{w}") / max(g(f"fd{t}_{w}"), 1e-9)
    return m


def block_ci(S, blocks, n_boot=1000, seed=0):
    """ブロック単位の再標本化で指標の 95% 区間を出す。"""
    rng = np.random.default_rng(seed)
    ub = sorted(set(blocks))
    by = {b: [x for x, bb in zip(S, blocks) if bb == b] for b in ub}
    keys = list(metrics(S))
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(len(ub), size=len(ub))
        r = metrics([x for i in pick for x in by[ub[i]]])
        for k in keys:
            acc[k].append(r[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in acc.items()}


def collect(days, scale=1.0):
    S, bad_cnt = [], 0
    for s in days:
        gs, gm, bad = load_day(s)
        S.append(day_stats(gs * scale, gm, bad))
    return S


if __name__ == "__main__":
    sets, block = make_split()
    # 学習集合で、GSM の総量を GSMaP に合わせる係数（総量比の逆数）を1つ決める
    tr = collect(sets["train"])
    scale = sum(x["true"] for x in tr) / sum(x["pred"] for x in tr)
    print(f"学習集合の総量比（GSMaP/GSM）: {scale:.3f}（これを掛けた基準も併記）\n")
    names = {"GSMそのまま": 1.0, f"GSM×{scale:.3f}": scale}
    for sname in ("train", "val"):
        days = sets[sname]
        blocks = [block[s] for s in days]
        print(f"=== {sname}（{len(days)}日・{len(set(blocks))}ブロック）===")
        res = {}
        for nm, sc in names.items():
            S = collect(days, sc)
            res[nm] = (metrics(S), block_ci(S, blocks, n_boot=500))
        keys = list(next(iter(res.values()))[0])
        print(f"{'指標':<12}" + "".join(f"{nm:>34}" for nm in names))
        for k in keys:
            row = f"{k:<12}"
            for nm in names:
                m, ci = res[nm]
                row += f"{m[k]:>12.3f} [{ci[k][0]:.3f},{ci[k][1]:.3f}]".rjust(34)
            print(row)
        print()
