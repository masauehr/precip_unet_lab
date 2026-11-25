"""Phase 1f: 量の損失を「平均を狙う」型に替える（対数MSEは中央値寄り）。

すべて乗法＋加法＋近傍のハイブリッド（Phase 1e）で、損失だけを変える:
  対数MSE        : 2段（発生BCE＋雨のある画素の対数MSE）… Phase 1e
  ガンマ偏差     : 2段（発生BCE＋雨のある画素のガンマ偏差）。期待値＝量×発生確率
  Tweedie(1.5)   : 単段（全画素・0を含む）。μ をそのまま予測とする
さらに、発生確率を持つ2つについて、地図（量×(確率>τ)）のしきい値 τ を 0.3/0.5/0.7 で比べる。

使い方: .venv/bin/python src/phase1f.py [--seeds 3] [--epochs 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import HybridCorrector, gamma_deviance_loss, tweedie_loss, two_stage_loss
from phase1b import metrics
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N, bootstrap_ci

LOSSES = ["対数MSE", "ガンマ偏差", "Tweedie"]


def train(loss_name, seed, epochs, n_train):
    torch.manual_seed(seed)
    c, f = make_dataset(n_train, N, FACTOR, seed=seed)
    xb = torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32)
    yb = torch.tensor(f[:, None], dtype=torch.float32)
    m = HybridCorrector(win=2 * FACTOR + 1, additive=True, neighbor=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), 16):
            idx = perm[i : i + 16]
            amount, logit = m.logits(xb[idx])
            if loss_name == "対数MSE":
                loss = two_stage_loss(amount, logit, yb[idx])
            elif loss_name == "ガンマ偏差":
                loss = gamma_deviance_loss(amount, logit, yb[idx])
            else:
                loss = tweedie_loss(amount, yb[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # 偏差損失は勾配が大きくなりやすい
            opt.step()
    return m.eval()


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

    rows = {}
    for s in range(a.seeds):
        c_te, f_te = make_dataset(a.n_test, N, FACTOR, seed=10_000 + s)
        b_te = regrid_conservative(c_te, (N, N))
        rows.setdefault("再格子化", []).append(metrics(b_te, f_te))
        for name in LOSSES:
            m = train(name, s, a.epochs, a.n_train)
            am, p = predict(m, b_te)
            if name == "Tweedie":
                rows.setdefault("Tweedie(単段)", []).append(metrics(am, f_te))
                continue
            rows.setdefault(f"{name}・期待値", []).append(metrics(am * p, f_te))
            for tau in (0.3, 0.5, 0.7):
                rows.setdefault(f"{name}・地図τ{tau}", []).append(metrics(am * (p > tau), f_te))
        print(f"シード {s} 完了", flush=True)

    print(f"\nシード{a.seeds}個 × テスト{a.n_test}件、エポック{a.epochs}、学習{a.n_train}件")
    print(f"{'設定':<18}{'対数RMSE':>22}{'総量比':>7}{'強雨比':>8}{'雨画素RMSE':>11}{'的中率':>7}{'空振り率':>8}")
    for name, ms in rows.items():
        r = np.concatenate([m["rmse"] for m in ms])
        mr, rl, rh = bootstrap_ci(r)
        g = lambda k: np.mean([m[k] for m in ms])
        print(f"{name:<16}{mr:7.3f} [{rl:.3f},{rh:.3f}]{g('total'):+7.2f}{g('strong'):+8.2f}"
              f"{g('wet_rmse'):11.3f}{g('pod'):7.2f}{g('far'):8.2f}")


if __name__ == "__main__":
    main()
