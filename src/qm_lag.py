"""QM更新の「遅れ」の検証（新しい期間・設定は結果を見る前に固定）。

qm_update.py では、直前の日までの真値が全部使えた（遅れ0）。実運用では真値（観測）が届くのに数日かかる。
更新のたびに、写像を作るのに使う履歴を「ブロック先頭の L 日前まで」に制限し、遅れ L=0・3・7・14 日、窓 N=30・60・90 日を全部報告する（選ばない）。
評価は、遅れ14日でも履歴が20日たまる日付順の34日目以降の共通の日。比べる: 固定QM と、遅れ付き更新QM（補正・基準とも）。
使い方: .venv/bin/python src/qm_lag.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import mean_stats
from final_fit import OUT, SEEDS
from fresh_eval import load_fresh
from phase5a import day_sets, paired_ci
from phase5c import apply_qm, fit_qm
from real_baseline import metrics
from real_model import make_model, predict

WINDOWS = (30, 60, 90)
LAGS = (0, 3, 7, 14)
START = 34
BLOCK = 10


def rolling_qm(P, Y, OK, n_days, window, lag):
    out = np.array(P)
    for b0 in range(START, n_days, BLOCK):
        hi = b0 - lag                     # 真値が届いているのは hi 日目の前まで
        lo = max(0, hi - window)
        sl = slice(lo * 24, hi * 24)
        m = fit_qm(P[sl], Y[sl], OK[sl])
        e = slice(b0 * 24, min(b0 + BLOCK, n_days) * 24)
        out[e] = apply_qm(m, P[e])
    return out


def main():
    days, x, y, bad = load_fresh()
    n = len(days)
    X, Y, OK = x.numpy()[:, 0], y.numpy()[:, 0], ~bad.numpy()[:, 0]
    C = pickle.load(open(OUT / "calib.pkl", "rb"))
    ev = slice(START * 24, n * 24)
    nev = n - START
    blocks = [(i - START) // BLOCK for i in range(START, n)]
    ys, bs = y[ev], bad[ev]
    stats = lambda P: day_sets(torch.from_numpy(P[ev][:, None]), ys, bs, nev)
    preds = {}
    for s in SEEDS:
        m = make_model()
        m.load_state_dict(torch.load(OUT / f"hybrid_seed{s}.pt"))
        preds[s] = predict(m.eval(), x).numpy()[:, 0]
    R = {"補正+固定QM": mean_stats([stats(apply_qm(C["qm_model"][s], preds[s])) for s in SEEDS]),
         "基準+固定QM": stats(apply_qm(C["qm_base"], X))}
    for w in WINDOWS:
        for lag in LAGS:
            R[f"補正 N{w} 遅れ{lag}"] = mean_stats([stats(rolling_qm(preds[s], Y, OK, n, w, lag)) for s in SEEDS])
            R[f"基準 N{w} 遅れ{lag}"] = stats(rolling_qm(X, Y, OK, n, w, lag))
        print(f"窓{w} 完了", flush=True)
    print(f"新しい期間 {n}日のうち {START}日目以降の {nev}日・{len(set(blocks))}ブロックで評価")
    show = ["対数RMSE", "総量比", "頻度比5", "CSI1", "CSI5", "FAR5"]
    for tag in ("補正", "基準"):
        print(f"\n■ {tag}（行: 方法 / 列: 指標）")
        print(f"{'方法':<18}" + "".join(f"{k:>10}" for k in show))
        names = [f"{tag}+固定QM"] + [f"{tag} N{w} 遅れ{l}" for w in WINDOWS for l in LAGS]
        for nm in names:
            m = metrics(R[nm])
            print(f"{nm:<18}" + "".join(f"{m[k]:>10.3f}" for k in show))
    print("\n■ 更新QM − 固定QM（補正・ブロック単位の対の差・* は区間が0を含まない）対数RMSE / 総量比")
    for w in WINDOWS:
        for lag in LAGS:
            d = paired_ci(R[f"補正 N{w} 遅れ{lag}"], R["補正+固定QM"], blocks)
            f = lambda k: f"{d[k][0]:+.3f}[{d[k][1]:+.3f},{d[k][2]:+.3f}]" + ("*" if d[k][1] > 0 or d[k][2] < 0 else "")
            print(f"  N{w:>2} 遅れ{lag:>2}: 対数RMSE {f('対数RMSE')}   総量比 {f('総量比')}")
    print("\n■ 遅れ14日 − 遅れ0日（補正・同じ窓）の対の差（遅れの代償）")
    for w in WINDOWS:
        d = paired_ci(R[f"補正 N{w} 遅れ14"], R[f"補正 N{w} 遅れ0"], blocks)
        f = lambda k: f"{d[k][0]:+.3f}[{d[k][1]:+.3f},{d[k][2]:+.3f}]" + ("*" if d[k][1] > 0 or d[k][2] < 0 else "")
        print(f"  N{w:>2}: 対数RMSE {f('対数RMSE')}   総量比 {f('総量比')}   頻度比5 {f('頻度比5')}")


if __name__ == "__main__":
    main()
