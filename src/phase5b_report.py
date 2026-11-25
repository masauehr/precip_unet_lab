"""Phase 5b の集計: 全分割をつなぎ、補正モデルと基準を（OOF=学習に使っていない状態で）比べる。

ブロック単位のブートストラップで、基準との対の差の 95% 区間を出す。学習シードごとの結果と、シード平均の差も出す。
使い方: .venv/bin/python src/phase5b_report.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5a import SHOW, paired_ci
from phase5b import OUT
from real_baseline import block_ci, metrics


def load(seed):
    S, B, base, days = [], [], [], []
    for f in sorted(OUT.glob(f"seed{seed}_fold*.pkl")):
        r = pickle.load(open(f, "rb"))
        S += r["model"]; base += r["base"]; B += r["blocks"]; days += r["days"]
    return S, base, B, days


if __name__ == "__main__":
    seeds = sorted({int(p.name.split("_")[0][4:]) for p in OUT.glob("seed*_fold*.pkl")})
    data = {s: load(s) for s in seeds}
    for s in seeds:
        S, base, B, d = data[s]
        print(f"seed {s}: {len(d)}日・{len(set(B))}ブロック（OOF）")
    S0, base0, B0, _ = data[seeds[0]]
    mb = metrics(base0)
    print(f"\n{'指標':<10}{'基準':>9}" + "".join(f"{'seed' + str(s):>9}" for s in seeds) + f"{'平均差':>9}  seedごとの差の範囲 / 平均差の区間(seed" + str(seeds[0]) + ")")
    diffs = {s: paired_ci(data[s][0], data[s][1], data[s][2], n_boot=500) for s in seeds}
    for k in SHOW:
        row = f"{k:<10}{mb[k]:>9.3f}"
        for s in seeds:
            row += f"{metrics(data[s][0])[k]:>9.3f}"
        ds = [diffs[s][k][0] for s in seeds]
        m, lo, hi = diffs[seeds[0]][k]
        mark = "*" if (lo > 0 or hi < 0) else " "
        row += f"{np.mean(ds):>+9.3f}  [{min(ds):+.3f}, {max(ds):+.3f}] / [{lo:+.3f}, {hi:+.3f}] {mark}"
        print(row)
    print("\n* = そのシードで、ブロック単位の差の区間が0を含まない")
