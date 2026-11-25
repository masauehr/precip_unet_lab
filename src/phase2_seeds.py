"""Phase 2 補足: 学習シードによるぶれの確認。既定・最大プーリング・power1.2 を、シード0〜4でシードごとに出す。

差がシード間のぶれより大きいか、シードごとの並び（対）が一貫しているかを見る。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1b import metrics
from phase2 import FACTOR, N, VARIANTS, fss, make_dataset, regrid_conservative, train

SEL = {"既定", "最大プーリング", "power1.2"}
torch.set_num_threads(4)
res = {}
for s in range(5):
    c, f = make_dataset(200, N, FACTOR, seed=10_000 + s)
    x = torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32)
    for name, kw, power in VARIANTS:
        if name not in SEL:
            continue
        m, _ = train(kw, power, s, 15, 400)
        with torch.no_grad():
            pred = m(x)[0].numpy()[:, 0]
        r = metrics(pred, f)
        res.setdefault(name, []).append((r["rmse"].mean(), fss(pred, f, 5.0), r["total"], r["wet_rmse"]))
        print(f"シード{s} {name:<10} 対数RMSE {r['rmse'].mean():.3f}  FSS5 {res[name][-1][1]:.3f}  総量比 {r['total']:+.2f}  雨画素RMSE {r['wet_rmse']:.3f}", flush=True)
print("\n5シードの 平均 ± 標準偏差（学習シード間）")
for name, v in res.items():
    a = np.array(v)
    print(f"  {name:<10} 対数RMSE {a[:,0].mean():.3f}±{a[:,0].std():.3f}  FSS5 {a[:,1].mean():.3f}±{a[:,1].std():.3f}"
          f"  総量比 {a[:,2].mean():+.2f}±{a[:,2].std():.2f}  雨画素RMSE {a[:,3].mean():.3f}±{a[:,3].std():.3f}")
base = np.array(res["既定"])
for name in ["最大プーリング", "power1.2"]:
    d = np.array(res[name]) - base
    print(f"  {name} − 既定（シードごとの差）対数RMSE {np.round(d[:,0],3).tolist()}  FSS5 {np.round(d[:,1],3).tolist()}")
