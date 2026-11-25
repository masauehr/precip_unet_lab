"""Phase 1b: 構造で総量過小に対処する（線形MSE・発生と量を分ける2段）。

比べる設定（評価はすべて ε=0.1 の対数RMSE・総量の偏り・強雨の比・雨あり画素の対数RMSE・的中率/空振り率）:
  再格子化          : 基準
  log_e0.1          : Phase 1 の対数MSE（参考）
  mse               : 線形MSE。条件付き平均を学ぶので総量は保たれやすい（強雨に引きずられる）
  2段_期待値        : 量(雨があるとき)×発生確率。期待値の予測
  2段_地図          : 量×(発生確率>0.5)。1枚の地図としての予測

使い方: .venv/bin/python src/phase1b.py [--seeds 3] [--epochs 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (WET, MultiplicativeCorrector, TwoStageCorrector, log_loss,
                   two_stage_loss)
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N, bootstrap_ci, log_rmse, mean_bias


def metrics(pred, true):
    """事例ごとの指標と、全体の比（強雨・的中率・空振り率）。"""
    strong = true >= 5.0
    wet_t, wet_p = true >= WET, pred >= WET
    d = (np.log(pred + 0.1) - np.log(true + 0.1)) ** 2
    return dict(
        rmse=log_rmse(pred, true),
        bias=mean_bias(pred, true),
        total=np.log(max(pred.sum(), 1e-9) / true.sum()),  # 全体の総量比（事例をまとめた値）
        strong=np.log(max(pred[strong].mean(), 1e-9) / true[strong].mean()),
        wet_rmse=np.sqrt(d[wet_t].mean()),
        pod=(wet_p & wet_t).sum() / wet_t.sum(),  # 雨のある画素を当てた割合
        far=(wet_p & ~wet_t).sum() / max(wet_p.sum(), 1),  # 予測した雨のうち外れた割合
    )


def train_one(kind, seed, epochs, n_train):
    torch.manual_seed(seed)
    c_tr, f_tr = make_dataset(n_train, N, FACTOR, seed=seed)
    t = lambda a: torch.tensor(a[:, None], dtype=torch.float32)
    xb, yb = t(regrid_conservative(c_tr, (N, N))), t(f_tr)
    model = TwoStageCorrector() if kind == "two" else MultiplicativeCorrector()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), 16):
            idx = perm[i : i + 16]
            if kind == "two":
                amount, logit = model.logits(xb[idx])
                loss = two_stage_loss(amount, logit, yb[idx])
            elif kind == "mse":
                loss = F.mse_loss(model(xb[idx]), yb[idx])
            else:
                loss = log_loss(model(xb[idx]), yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)
    t = lambda x: torch.tensor(x[:, None], dtype=torch.float32)

    rows = {k: [] for k in ["再格子化", "log_e0.1", "mse", "2段_期待値", "2段_地図"]}
    for s in range(a.seeds):
        c, f = make_dataset(a.n_test, N, FACTOR, seed=10_000 + s)
        xt = t(regrid_conservative(c, (N, N)))
        rows["再格子化"].append(metrics(xt.numpy()[:, 0], f))
        for kind, name in [("log", "log_e0.1"), ("mse", "mse")]:
            m = train_one(kind, s, a.epochs, a.n_train)
            with torch.no_grad():
                rows[name].append(metrics(m(xt).numpy()[:, 0], f))
        m = train_one("two", s, a.epochs, a.n_train)
        with torch.no_grad():
            amount, p = m(xt)
        rows["2段_期待値"].append(metrics((amount * p).numpy()[:, 0], f))
        rows["2段_地図"].append(metrics((amount * (p > 0.5)).numpy()[:, 0], f))
        print(f"シード {s} 完了", flush=True)

    print(f"\nシード{a.seeds}個 × テスト{a.n_test}件、エポック{a.epochs}、学習{a.n_train}件")
    print(f"{'設定':<12}{'対数RMSE':>22}{'事例ごとの総量の偏り':>24}{'総量比':>7}{'強雨比':>8}{'雨画素RMSE':>11}{'的中率':>7}{'空振り率':>8}")
    for name, ms in rows.items():
        r = np.concatenate([m["rmse"] for m in ms])
        b = np.concatenate([m["bias"] for m in ms])
        mr, rl, rh = bootstrap_ci(r)
        mb, bl, bh = bootstrap_ci(b)
        g = lambda k: np.mean([m[k] for m in ms])
        print(f"{name:<10}{mr:7.3f} [{rl:.3f},{rh:.3f}] {mb:+7.3f} [{bl:+.3f},{bh:+.3f}]"
              f"{g('total'):+7.2f}{g('strong'):+8.2f}{g('wet_rmse'):11.3f}{g('pod'):7.2f}{g('far'):8.2f}")


if __name__ == "__main__":
    main()
