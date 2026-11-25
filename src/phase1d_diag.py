"""Phase 1d 診断: 乗法補正は「ベース場が0なら0」。真値に雨があるのにベース場が小さい画素がどれだけあるか。

(1) 学習なしの数え上げ: 雨のある画素のうち、ベース場が WET 未満の割合と、その真値の総量の割合。
(2) 学習した2段で、μ をベース場の大きさ別に分けて見る。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS, WET
from phase1d import mu_sigma, predict, train
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N

c, f = make_dataset(200, N, FACTOR, seed=10_000)
base = regrid_conservative(c, (N, N))
wet = f >= WET
low = base < WET
print("[数え上げ] テスト200件")
print(f"  雨のある画素: {wet.sum()} / {wet.size}（{wet.mean():.1%}）")
print(f"  そのうちベース場が {WET} 未満: {(wet & low).sum() / wet.sum():.1%}（量を出せない画素）")
print(f"  真値の総量のうち、その画素の割合: {f[wet & low].sum() / f.sum():.1%}")
strong = f >= 5.0
print(f"  強雨(≥5mm/h)画素のうちベース場が {WET} 未満: {(strong & low).sum() / strong.sum():.1%}、"
      f"ベース場が真値の1/2未満: {(strong & (base < f / 2)).sum() / strong.sum():.1%}")

m, _, _ = train(15, 400, 0.0, 0)
x = torch.tensor(base[:, None], dtype=torch.float32)
am, pr = predict(m, x)
print("\n[μ をベース場の大きさ別に] 15エポック・400件")
for lo, hi, name in [(0, WET, "ベース<0.1"), (WET, 1.0, "0.1〜1"), (1.0, 5.0, "1〜5"), (5.0, 1e9, "≥5")]:
    sel = wet & (base >= lo) & (base < hi)
    if sel.sum() == 0:
        continue
    d = np.log(f[sel] + EPS) - np.log(am[sel] + EPS)
    print(f"  {name:<10} 画素数{sel.sum():7d}  μ={d.mean():+.2f}  σ²={d.var():.2f}")
d = np.log(f[wet & ~low] + EPS) - np.log(am[wet & ~low] + EPS)
print(f"\n  ベース≥0.1 の雨画素だけの μ={d.mean():+.2f}（全体の μ と比べる）")
