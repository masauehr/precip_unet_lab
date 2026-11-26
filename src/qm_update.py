"""QM の更新運用の検証（新しい期間 2026-05-11〜・設定は結果を見る前に固定）。

問い: 固定した QM（297日のOOFで作成）は期間が変わると総量・頻度が外れた。**直近の期間で写像を作り直す**と改善するか。
比べる方法（補正+QM と 基準+QM の両方）:
  固定QM / 更新QM（直近 N 日で作成・N=30,60,90 を全部報告・N は選ばない）
更新の決まり: 10日ブロックの先頭で、直前 N 日（新期間内。履歴が N 日未満なら有る分だけ）の（予測, 真値）から写像を作り直し、そのブロックに適用。
  真値は直前の日までのものだけを使う（運用で使える情報だけ）。評価は履歴が20日たまった後（日付順の20日目以降）で、すべての方法に共通。
モデルは変更しない（最終モデル）。使い方: .venv/bin/python src/qm_update.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import mean_stats
from final_fit import OUT, SEEDS
import pickle
from fresh_eval import load_fresh
from phase5a import SHOW, day_sets, paired_ci
from phase5c import apply_qm, fit_qm
from real_baseline import metrics
from real_model import make_model, predict

WINDOWS = (30, 60, 90)
START = 20          # 評価の開始（日付順の日数）
BLOCK = 10


def rolling_qm(P, Y, OK, n_days, window):
    """ブロックごとに、直前 window 日で写像を作り直して適用する（評価は START 日目以降のブロックのみ）。"""
    out = np.array(P)
    for b0 in range(START, n_days, BLOCK):
        lo = max(0, b0 - window)
        sl = slice(lo * 24, b0 * 24)
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
    R = {}
    R["基準（QMなし）"] = stats(X)
    R["基準+固定QM"] = stats(apply_qm(C["qm_base"], X))
    for w in WINDOWS:
        R[f"基準+更新QM{w}"] = stats(rolling_qm(X, Y, OK, n, w))
    R["補正+固定QM"] = mean_stats([stats(apply_qm(C["qm_model"][s], preds[s])) for s in SEEDS])
    for w in WINDOWS:
        R[f"補正+更新QM{w}"] = mean_stats([stats(rolling_qm(preds[s], Y, OK, n, w)) for s in SEEDS])
    print(f"新しい期間 {n}日のうち、履歴{START}日後の {nev}日・{len(set(blocks))}ブロックで評価")
    rows = {k: metrics(v) for k, v in R.items()}
    show = ["対数RMSE", "総量比", "CSI1", "頻度比1", "CSI5", "FAR5", "頻度比5", "CSI10", "頻度比10"]
    print(f"{'指標':<10}" + "".join(f"{k:>18}" for k in rows))
    for k in show:
        print(f"{k:<10}" + "".join(f"{r[k]:>18.3f}" for r in rows.values()))
    for tag in ("基準", "補正"):
        for w in WINDOWS:
            d = paired_ci(R[f"{tag}+更新QM{w}"], R[f"{tag}+固定QM"], blocks)
            print(f"\n{tag}+更新QM{w} − {tag}+固定QM（ブロック単位・対の差・* は区間が0を含まない）")
            for k in show:
                mm, lo, hi = d[k]
                print(f"  {k:<10}{mm:>+9.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if (lo > 0 or hi < 0) else ""))
    # 追加解析（結果を見たあとに加えた・参考）: 更新QMどうしで、補正モデルの上乗せがあるか（頻度がほぼそろうので公平）
    for w in WINDOWS:
        d = paired_ci(R[f"補正+更新QM{w}"], R[f"基準+更新QM{w}"], blocks)
        print(f"\n【追加解析】補正+更新QM{w} − 基準+更新QM{w}（ブロック単位・対の差）")
        for k in show:
            mm, lo, hi = d[k]
            print(f"  {k:<10}{mm:>+9.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if (lo > 0 or hi < 0) else ""))


if __name__ == "__main__":
    main()
