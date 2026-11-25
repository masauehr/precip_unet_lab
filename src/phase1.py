"""Phase 1: 対数損失で総量が落ちる問題への対処を、複数シードで比べる。

比べる設定（評価は常に ε=0.1 の対数RMSEと、総量の偏りで統一）:
  log_e0.1      : 対数MSE(ε=0.1)                       … Phase 0 の設定
  log_e1.0      : 対数MSE(ε=1.0)                       … 弱雨・ゼロ付近を軽くみる
  log+total     : 対数MSE + 粗い平均を真値に合わせる損失  … 総量を直接罰する
  log_e0.1+renorm: 学習後に粗い格子平均を入力の粗い予報に再スケール … 入力の総量を保つ（入力の偏りは残る）

使い方: .venv/bin/python src/phase1.py [--seeds 3] [--epochs 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MultiplicativeCorrector, coarse_total_loss, log_loss, renormalize
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N, bootstrap_ci, log_rmse, mean_bias

CONFIGS = {
    "log_e0.1": dict(eps=0.1, w_total=0.0, renorm=False),
    "log_e1.0": dict(eps=1.0, w_total=0.0, renorm=False),
    "log+total": dict(eps=0.1, w_total=1.0, renorm=False),
    "log_e0.1+renorm": dict(eps=0.1, w_total=0.0, renorm=True),
}


def run(cfg, seed, epochs, n_train, n_test):
    torch.manual_seed(seed)
    c_tr, f_tr = make_dataset(n_train, N, FACTOR, seed=seed)
    c_te, f_te = make_dataset(n_test, N, FACTOR, seed=10_000 + seed)
    t = lambda a: torch.tensor(a[:, None], dtype=torch.float32)
    xb, yb = t(regrid_conservative(c_tr, (N, N))), t(f_tr)
    model = MultiplicativeCorrector(base=16, depth=3)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 16
    for _ in range(epochs):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), bs):
            idx = perm[i : i + bs]
            pred = model(xb[idx])
            loss = log_loss(pred, yb[idx], cfg["eps"])
            if cfg["w_total"] > 0:
                loss = loss + cfg["w_total"] * coarse_total_loss(pred, yb[idx], FACTOR, cfg["eps"])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xt = t(regrid_conservative(c_te, (N, N)))
        pred = model(xt)
        if cfg["renorm"]:
            pred = renormalize(pred, t(c_te), FACTOR)
    pred = pred.numpy()[:, 0]
    strong = f_te >= 5.0  # 強雨（真値 5 mm/h 以上）の画素での予測/真値の比
    return (
        log_rmse(pred, f_te),
        mean_bias(pred, f_te),
        np.log(pred[strong].mean() / f_te[strong].mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)

    # 基準（保存的再格子化のみ）
    base_r, base_b = [], []
    for s in range(a.seeds):
        c, f = make_dataset(a.n_test, N, FACTOR, seed=10_000 + s)
        p = regrid_conservative(c, (N, N))
        base_r.append(log_rmse(p, f))
        base_b.append(mean_bias(p, f))
    base_r, base_b = np.concatenate(base_r), np.concatenate(base_b)

    print(f"シード{a.seeds}個 × テスト{a.n_test}件 = {a.seeds * a.n_test}件、エポック{a.epochs}、学習{a.n_train}件")
    print(f"{'設定':<18}{'対数RMSE':>20}{'総量の偏り(対数比)':>28}{'強雨(≥5)の比(対数)':>22}")
    m, lo, hi = bootstrap_ci(base_r)
    mb, blo, bhi = bootstrap_ci(base_b)
    print(f"{'基準(再格子化)':<16}{m:8.3f} [{lo:.3f},{hi:.3f}]  {mb:+7.3f} [{blo:+.3f},{bhi:+.3f}]")
    for name, cfg in CONFIGS.items():
        rs, bs_, st = [], [], []
        for s in range(a.seeds):
            r, b, sr = run(cfg, s, a.epochs, a.n_train, a.n_test)
            rs.append(r), bs_.append(b), st.append(sr)
        r, b = np.concatenate(rs), np.concatenate(bs_)
        m, lo, hi = bootstrap_ci(r)
        mb, blo, bhi = bootstrap_ci(b)
        print(
            f"{name:<18}{m:8.3f} [{lo:.3f},{hi:.3f}]  {mb:+7.3f} [{blo:+.3f},{bhi:+.3f}]"
            f"   {np.mean(st):+7.3f}±{np.std(st):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
