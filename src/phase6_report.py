"""Phase 6 の集計: 予報時間（時刻）を入力に加えたモデル（FT）と、Phase 5b のモデル（基準モデル）を OOF で比べる。

QM なし／あり（Phase 5c と同じ: 他分割の OOF で写像を作る）。シード3つの日別集計を平均して、ブロック単位の対の差（FT − 5b）を出す。
テスト・新しい期間は使わない。使い方: .venv/bin/python src/phase6_report.py [6|7]   （6: 時刻入力・7: 前後の時刻入力）
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import mean_stats
from phase5a import SHOW, day_sets, paired_ci
from phase5b import OUT as OUT_B, build
from phase5c import apply_qm, fit_qm
from phase6 import OUT as OUT_6
import phase7
from real_baseline import metrics
from real_model import load_set
from split import make_split

SEEDS = (0, 1, 2)
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "6"          # "6": 時刻(FT)入力  "7": 前後の時刻入力
OUT_V = OUT_6 if VARIANT == "6" else phase7.OUT
TAG = "FT" if VARIANT == "6" else "CTX"


def main():
    days, blocks, fold = build(6)
    fold_arr = np.repeat(np.array(fold), 24)
    sets = make_split()[0]
    x1, y1, b1 = load_set(sets["train"], "cache_train.npz")
    x2, y2, b2 = load_set(sets["val"], "cache_val.npz")
    x, y, bad = (torch.cat(p) for p in ((x1, x2), (y1, y2), (b1, b2)))
    X, Y, OK = x.numpy()[:, 0], y.numpy()[:, 0], ~bad.numpy()[:, 0]
    n = len(days)
    stats = lambda P: day_sets(torch.from_numpy(P[:, None]), y, bad, n)

    def load(d, pre, s):
        P = np.zeros_like(X)
        for k in range(6):
            P[fold_arr == k] = np.load(d / f"{pre}_seed{s}_fold{k}.npz")["pred"][:, 0].astype(np.float32)
        return P

    def qm(P):
        Q = np.zeros_like(P)
        for k in range(6):
            te, fit = fold_arr == k, fold_arr != k
            Q[te] = apply_qm(fit_qm(P[fit], Y[fit], OK[fit]), P[te])
        return Q

    R = {"基準": stats(X)}
    for tag, d in (("5b", OUT_B), (TAG, OUT_V)):
        raw, q = [], []
        for s in SEEDS:
            P = load(d, "pred", s)
            raw.append(stats(P)); q.append(stats(qm(P)))
            print(tag, "seed", s, "完了", flush=True)
        R[f"{tag}"], R[f"{tag}+QM"] = mean_stats(raw), mean_stats(q)
    print(f"\nOOF {n}日（シード{SEEDS}の日別集計の平均）")
    rows = {k: metrics(v) for k, v in R.items()}
    print(f"{'指標':<10}" + "".join(f"{k:>10}" for k in rows))
    for k in SHOW:
        print(f"{k:<10}" + "".join(f"{r[k]:>10.3f}" for r in rows.values()))
    for a_, b_ in ((TAG, "5b"), (f"{TAG}+QM", "5b+QM")):
        d = paired_ci(R[a_], R[b_], blocks)
        print(f"\n{a_} − {b_}（ブロック単位の対の差・* は区間が0を含まない）")
        for k in SHOW:
            m, lo, hi = d[k]
            print(f"  {k:<10}{m:>+9.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if (lo > 0 or hi < 0) else ""))


if __name__ == "__main__":
    main()
