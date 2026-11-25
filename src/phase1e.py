"""Phase 1e: 乗法＋加法のハイブリッド（ベース場が0の所にも雨を出す）。

比べる設定（すべて発生と量を分ける2段・期待値＝量×発生確率）:
  乗法のみ      : Phase 1b の2段（基準）
  +加法         : 量 = ベース場×exp(r) + softplus(a)
  +加法+近傍    : さらに近傍（窓17）の最大・平均を入力に加える
  （各行に「+分位点」＝Phase 1c の補正を加えた版も出す）

使い方: .venv/bin/python src/phase1e.py [--seeds 3] [--epochs 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS, WET, HybridCorrector, two_stage_loss
from phase1b import metrics
from phase1c import apply_quantile_map, fit_quantile_map
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N, bootstrap_ci

VARIANTS = {
    "乗法のみ": dict(additive=False, neighbor=False),
    "+加法": dict(additive=True, neighbor=False),
    "+加法+近傍": dict(additive=True, neighbor=True),
}


def train(cfg, seed, epochs, n_train):
    torch.manual_seed(seed)
    c, f = make_dataset(n_train, N, FACTOR, seed=seed)
    xb = torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32)
    yb = torch.tensor(f[:, None], dtype=torch.float32)
    m = HybridCorrector(win=2 * FACTOR + 1, **cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), 16):
            idx = perm[i : i + 16]
            amount, logit = m.logits(xb[idx])
            loss = two_stage_loss(amount, logit, yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval(), xb.numpy()[:, 0], f


def predict(m, base):
    with torch.no_grad():
        amount, p = m(torch.tensor(base[:, None], dtype=torch.float32))
    return amount.numpy()[:, 0], p.numpy()[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)

    rows, mus = {}, {}
    for s in range(a.seeds):
        c_te, f_te = make_dataset(a.n_test, N, FACTOR, seed=10_000 + s)
        b_te = regrid_conservative(c_te, (N, N))
        rows.setdefault("再格子化", []).append(metrics(b_te, f_te))
        for name, cfg in VARIANTS.items():
            m, b_tr, f_tr = train(cfg, s, a.epochs, a.n_train)
            am_tr, pr_tr = predict(m, b_tr)
            am_te, pr_te = predict(m, b_te)
            rows.setdefault(name, []).append(metrics(am_te * pr_te, f_te))
            qp, qt = fit_quantile_map(am_tr * pr_tr, f_tr)
            rows.setdefault(name + "+分位点", []).append(
                metrics(apply_quantile_map(am_te * pr_te, qp, qt), f_te))
            # ベース場が薄い（<0.1）雨画素での、量の対数残差の平均 μ（テスト）
            sel = (f_te >= WET) & (b_te < WET)
            mus.setdefault(name, []).append(
                (np.log(f_te[sel] + EPS) - np.log(am_te[sel] + EPS)).mean())
        print(f"シード {s} 完了", flush=True)

    print(f"\nシード{a.seeds}個 × テスト{a.n_test}件、エポック{a.epochs}、学習{a.n_train}件")
    print(f"{'設定':<16}{'対数RMSE':>22}{'総量比':>7}{'強雨比':>8}{'雨画素RMSE':>11}{'的中率':>7}{'空振り率':>8}")
    for name, ms in rows.items():
        r = np.concatenate([m["rmse"] for m in ms])
        mr, rl, rh = bootstrap_ci(r)
        g = lambda k: np.mean([m[k] for m in ms])
        print(f"{name:<14}{mr:7.3f} [{rl:.3f},{rh:.3f}]{g('total'):+7.2f}{g('strong'):+8.2f}"
              f"{g('wet_rmse'):11.3f}{g('pod'):7.2f}{g('far'):8.2f}")
    print("\nベース場<0.1 の雨画素での量の対数残差の平均 μ（0に近いほど良い）")
    for name, v in mus.items():
        print(f"  {name:<14}{np.mean(v):+.2f} ± {np.std(v):.2f}")


if __name__ == "__main__":
    main()
