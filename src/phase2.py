"""Phase 2: アブレーション（既定から1つずつ変える）。

既定: 乗法＋加法＋近傍のハイブリッド・Tweedie(power=1.5)単段・U-Net(深さ3・GroupNorm・平均プーリング・
      スキップ結合・双線形拡大)。各設定を複数シードで学習し、同じテストで比べる。
指標: 対数RMSE・総量比・強雨比・雨画素RMSE・的中率・空振り率に、位置のずれに強い FSS（1・5 mm/h、窓9）を追加。

使い方: .venv/bin/python src/phase2.py [--seeds 3] [--epochs 15]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import HybridCorrector, tweedie_loss
from phase1b import metrics
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N, bootstrap_ci, fss

# (名前, モデルの引数, Tweedie の power)
DEFAULT = dict(additive=True, neighbor=True, depth=3, norm="group", pool="avg", skip=True, up="bilinear")
VARIANTS = [
    ("既定", {}, 1.5),
    ("正規化なし", dict(norm="none"), 1.5),
    ("最大プーリング", dict(pool="max"), 1.5),
    ("スキップなし", dict(skip=False), 1.5),
    ("最近傍拡大", dict(up="nearest"), 1.5),
    ("深さ2", dict(depth=2), 1.5),
    ("深さ4", dict(depth=4), 1.5),
    ("加法なし(乗法のみ)", dict(additive=False), 1.5),
    ("近傍なし", dict(neighbor=False), 1.5),
    ("power1.2", {}, 1.2),
    ("power1.8", {}, 1.8),
]


def train(kw, power, seed, epochs, n_train):
    torch.manual_seed(seed)
    c, f = make_dataset(n_train, N, FACTOR, seed=seed)
    xb = torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32)
    yb = torch.tensor(f[:, None], dtype=torch.float32)
    m = HybridCorrector(win=2 * FACTOR + 1, **{**DEFAULT, **kw})
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), 16):
            idx = perm[i : i + 16]
            amount, _ = m.logits(xb[idx])
            loss = tweedie_loss(amount, yb[idx], power)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m.eval(), sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)

    tests = []
    for s in range(a.seeds):
        c, f = make_dataset(a.n_test, N, FACTOR, seed=10_000 + s)
        tests.append((torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32), f))

    hdr = (f"{'設定':<16}{'対数RMSE':>22}{'総量比':>7}{'強雨比':>8}{'雨画素RMSE':>11}"
           f"{'的中率':>7}{'空振り率':>8}{'FSS1':>7}{'FSS5':>7}{'パラメータ':>10}{'秒/シード':>9}")
    print(f"シード{a.seeds}個 × テスト{a.n_test}件、エポック{a.epochs}、学習{a.n_train}件（既定から1つずつ変更）")
    print(hdr, flush=True)
    for name, kw, power in VARIANTS:
        ms, f1, f5, t0 = [], [], [], time.time()
        for s in range(a.seeds):
            m, npar = train(kw, power, s, a.epochs, a.n_train)
            x, f = tests[s]
            with torch.no_grad():
                pred = m(x)[0].numpy()[:, 0]
            ms.append(metrics(pred, f))
            f1.append(fss(pred, f, 1.0))
            f5.append(fss(pred, f, 5.0))
        r = np.concatenate([m["rmse"] for m in ms])
        mr, rl, rh = bootstrap_ci(r)
        g = lambda k: np.mean([m[k] for m in ms])
        print(f"{name:<14}{mr:7.3f} [{rl:.3f},{rh:.3f}]{g('total'):+7.2f}{g('strong'):+8.2f}"
              f"{g('wet_rmse'):11.3f}{g('pod'):7.2f}{g('far'):8.2f}{np.mean(f1):7.3f}{np.mean(f5):7.3f}"
              f"{npar:10d}{(time.time() - t0) / a.seeds:9.0f}", flush=True)


if __name__ == "__main__":
    main()
