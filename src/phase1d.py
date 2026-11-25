"""Phase 1d: (1) 量が学習不足か、(2) 強雨に重みを付けると強雨が戻るか。

(1) エポック数・学習量を変えて、雨のある画素の対数残差の平均 μ（学習・テスト）とテストの指標を見る。
    μ>0 は、対数空間でも量が過小（＝まだ学習できる余地、または表現力不足）。
(2) 量の損失で、真値5mm/h以上の画素に重みを付ける（w_strong）。

使い方: .venv/bin/python src/phase1d.py [--seed 0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS, WET, TwoStageCorrector, two_stage_loss
from phase1b import metrics
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N

# (名前, エポック, 学習件数, 強雨の重み)
CONFIGS = [
    ("e15_n400", 15, 400, 0.0),
    ("e45_n400", 45, 400, 0.0),
    ("e45_n1200", 45, 1200, 0.0),
    ("e45_n400_強雨x5", 45, 400, 4.0),
]


def train(epochs, n_train, w_strong, seed):
    torch.manual_seed(seed)
    c, f = make_dataset(n_train, N, FACTOR, seed=seed)
    xb = torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32)
    yb = torch.tensor(f[:, None], dtype=torch.float32)
    m = TwoStageCorrector()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), 16):
            idx = perm[i : i + 16]
            amount, logit = m.logits(xb[idx])
            loss = two_stage_loss(amount, logit, yb[idx], w_strong=w_strong)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval(), xb, f


def predict(m, x):
    with torch.no_grad():
        amount, p = m(x)
    return amount.numpy()[:, 0], p.numpy()[:, 0]


def mu_sigma(amount, true):
    wet = true >= WET
    d = np.log(true[wet] + EPS) - np.log(amount[wet] + EPS)
    return d.mean(), d.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)
    c_te, f_te = make_dataset(a.n_test, N, FACTOR, seed=10_000 + a.seed)
    x_te = torch.tensor(regrid_conservative(c_te, (N, N))[:, None], dtype=torch.float32)

    print(f"{'設定':<18}{'秒':>5}{'μ学習':>8}{'μテスト':>8}{'σ²テスト':>9}{'対数RMSE':>9}{'総量比':>7}{'強雨比':>8}{'雨画素RMSE':>11}{'的中率':>7}", flush=True)
    for name, ep, n, w in CONFIGS:
        t0 = time.time()
        m, x_tr, f_tr = train(ep, n, w, a.seed)
        am_tr, _ = predict(m, x_tr)
        am_te, pr_te = predict(m, x_te)
        mu_tr, _ = mu_sigma(am_tr, f_tr)
        mu_te, var_te = mu_sigma(am_te, f_te)
        r = metrics(am_te * pr_te, f_te)
        print(
            f"{name:<16}{time.time() - t0:5.0f}{mu_tr:+8.2f}{mu_te:+8.2f}{var_te:9.2f}"
            f"{r['rmse'].mean():9.3f}{r['total']:+7.2f}{r['strong']:+8.2f}{r['wet_rmse']:11.3f}{r['pod']:7.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
