"""合成データで U-Net（乗法補正）を学習し、基準と比べる。

使い方: .venv/bin/python src/train.py [--epochs 30] [--seed 0]

比較する3つ（すべて目標格子 64x64 上、対数空間 RMSE と平均バイアスで評価）:
  1. 保存的再格子化のみ（ベース場そのもの）
  2. 双線形補間のみ（総量が保存されない基準）
  3. 保存的再格子化 + U-Net 乗法補正
評価の信頼区間は、事例を単位としたブートストラップで出す。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS, MultiplicativeCorrector, conservation_loss, log_loss
from regrid import regrid_bilinear, regrid_conservative
from synth import make_dataset

FACTOR = 8
N = 64


def log_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """事例ごとの対数空間 RMSE (N,)。"""
    d = np.log(pred + EPS) - np.log(true + EPS)
    return np.sqrt((d**2).mean(axis=(-2, -1)))


def mean_bias(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """事例ごとの面積平均降水量の比 pred/true の対数（0 なら偏りなし、無降水の事例は 0）。"""
    p, t = pred.mean(axis=(-2, -1)), true.mean(axis=(-2, -1))
    ok = t > 1e-6
    out = np.zeros_like(t)
    # 予測が全面0の事例で -inf にならないよう下限を置く（1e-3 mm/h）
    out[ok] = np.log(np.maximum(p[ok], 1e-3) / t[ok])
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, block: int = 1):
    """平均のブートストラップ 95% 信頼区間。block>1 なら連続 block 件のまとまりで再標本化する。"""
    rng = np.random.default_rng(seed)
    n = len(values)
    nb = max(n // block, 1)
    starts_all = np.arange(0, nb * block, block)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.choice(starts_all, size=nb)
        idx = (starts[:, None] + np.arange(block)).ravel()
        means[i] = values[idx].mean()
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, **kw):
    """事例ごとの差 a-b の平均と信頼区間。0 を含まなければ差は誤差でないとみなす。"""
    return bootstrap_ci(a - b, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w_cons", type=float, default=0.0, help="保存制約損失の重み")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    nc = N // FACTOR
    # 学習用とテスト用は別の乱数列（シード差）で作る
    c_tr, f_tr = make_dataset(args.n_train, N, FACTOR, seed=args.seed)
    c_te, f_te = make_dataset(args.n_test, N, FACTOR, seed=10_000 + args.seed)

    base_tr = regrid_conservative(c_tr, (N, N))
    base_te = regrid_conservative(c_te, (N, N))
    bil_te = regrid_bilinear(c_te, (N, N))

    t = lambda a: torch.tensor(a[:, None], dtype=torch.float32)
    xb, yb, cb = t(base_tr), t(f_tr), t(c_tr)

    model = MultiplicativeCorrector(base=16, depth=3)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 16
    for ep in range(args.epochs):
        perm = torch.randperm(len(xb))
        tot = 0.0
        for i in range(0, len(xb), bs):
            idx = perm[i : i + bs]
            pred = model(xb[idx])
            loss = log_loss(pred, yb[idx])
            if args.w_cons > 0:
                loss = loss + args.w_cons * conservation_loss(pred, cb[idx], FACTOR)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep + 1:3d}  学習損失(対数MSE) {tot / len(xb):.4f}")

    model.eval()
    with torch.no_grad():
        unet_te = model(t(base_te)).numpy()[:, 0]

    res = {
        "保存的再格子化のみ": base_te,
        "双線形のみ": bil_te,
        "保存的+U-Net乗法補正": unet_te,
    }
    print(f"\nテスト {args.n_test} 件（事例単位のブートストラップ 95%信頼区間）")
    print(f"{'手法':<22}{'対数RMSE':>24}{'面積平均の偏り(対数比)':>34}")
    scores = {}
    for name, p in res.items():
        r = log_rmse(p, f_te)
        b = mean_bias(p, f_te)
        scores[name] = r
        m, lo, hi = bootstrap_ci(r)
        mb, blo, bhi = bootstrap_ci(b)
        print(f"{name:<20}{m:8.3f} [{lo:.3f}, {hi:.3f}]   {mb:+8.3f} [{blo:+.3f}, {bhi:+.3f}]")

    d, lo, hi = paired_diff_ci(scores["保存的+U-Net乗法補正"], scores["保存的再格子化のみ"])
    verdict = "差は誤差でない" if (hi < 0 or lo > 0) else "差は誤差の範囲"
    print(f"\nU-Net − 基準（対数RMSEの差、負なら改善）: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] → {verdict}")


if __name__ == "__main__":
    main()
