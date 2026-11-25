"""Phase 1c: 総量の偏りを、学習後の別段で補正する。

学習データで補正を求め、テストに適用する（テストの真値は使わない）。
  倍率      : 学習データで sum(真値)/sum(予測) を求め、全体に掛ける
  対数正規  : 量を対数損失で学ぶと中央値寄りになる、という見立ての補正。
              学習データの雨のある画素で残差 d=log(真+ε)-log(量+ε) の平均 μ・分散 σ² を求め、量×exp(μ+σ²/2)
  分位点写像: 学習データで「予測の分布」を「真値の分布」に写す（降水の標準的なバイアス補正）

使い方: .venv/bin/python src/phase1c.py [--seeds 3] [--epochs 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS, WET
from phase1b import metrics, train_one
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N, bootstrap_ci

NQ = 2000  # 分位点の数


def fit_scale(pred, true):
    return true.sum() / max(pred.sum(), 1e-9)


def fit_quantile_map(pred, true):
    """予測の分位点 → 真値の分位点。雨のない画素も含めた全画素で作る。"""
    q = np.linspace(0, 1, NQ)
    # 上位は細かく取る（強雨の裾を拾う）
    q = np.unique(np.concatenate([q, 1 - np.logspace(-6, -3, 200)]))
    return np.quantile(pred, q), np.quantile(true, q)


def apply_quantile_map(x, qp, qt):
    # 予測が同値で並ぶ（0が大量）箇所は interp が安定するよう微小に単調化
    qp = qp + np.arange(len(qp)) * 1e-9
    return np.interp(x, qp, qt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)
    t = lambda x: torch.tensor(x[:, None], dtype=torch.float32)

    rows = {}
    add = lambda name, m: rows.setdefault(name, []).append(m)
    diag = []
    for s in range(a.seeds):
        c_tr, f_tr = make_dataset(a.n_train, N, FACTOR, seed=s)
        c_te, f_te = make_dataset(a.n_test, N, FACTOR, seed=10_000 + s)
        x_tr, x_te = t(regrid_conservative(c_tr, (N, N))), t(regrid_conservative(c_te, (N, N)))

        # 基準（再格子化）
        b_tr, b_te = x_tr.numpy()[:, 0], x_te.numpy()[:, 0]
        add("再格子化", metrics(b_te, f_te))
        add("再格子化+倍率", metrics(b_te * fit_scale(b_tr, f_tr), f_te))

        # 線形MSE
        m = train_one("mse", s, a.epochs, a.n_train)
        with torch.no_grad():
            p_tr, p_te = m(x_tr).numpy()[:, 0], m(x_te).numpy()[:, 0]
        add("線形MSE", metrics(p_te, f_te))
        add("線形MSE+倍率", metrics(p_te * fit_scale(p_tr, f_tr), f_te))
        qp, qt = fit_quantile_map(p_tr, f_tr)
        add("線形MSE+分位点", metrics(apply_quantile_map(p_te, qp, qt), f_te))

        # 2段（期待値）
        m = train_one("two", s, a.epochs, a.n_train)
        with torch.no_grad():
            am_tr, pr_tr = m(x_tr)
            am_te, pr_te = m(x_te)
        am_tr, pr_tr, am_te, pr_te = (v.numpy()[:, 0] for v in (am_tr, pr_tr, am_te, pr_te))
        e_tr, e_te = am_tr * pr_tr, am_te * pr_te
        add("2段", metrics(e_te, f_te))
        add("2段+倍率", metrics(e_te * fit_scale(e_tr, f_tr), f_te))
        # 対数正規の補正（雨のある画素の残差から）
        wet = f_tr >= WET
        d = np.log(f_tr[wet] + EPS) - np.log(am_tr[wet] + EPS)
        mu, var = d.mean(), d.var()
        corr = np.exp(mu + var / 2)
        add("2段+対数正規", metrics(am_te * corr * pr_te, f_te))
        qp, qt = fit_quantile_map(e_tr, f_tr)
        add("2段+分位点", metrics(apply_quantile_map(e_te, qp, qt), f_te))
        actual = f_tr[wet].sum() / max(am_tr[wet].sum(), 1e-9)  # 雨のある画素の実際の総量比
        diag.append((mu, var, corr, actual))
        print(f"シード {s} 完了", flush=True)

    dg = np.array(diag)
    print("\n[見立ての確認] 雨のある画素の量（学習データ）")
    print(f"  対数残差の平均μ={dg[:,0].mean():+.3f}（0に近い＝対数では偏りなし）、分散σ²={dg[:,1].mean():.3f}")
    print(f"  対数正規の補正倍率 exp(μ+σ²/2)={dg[:,2].mean():.2f}、実際の総量比 真/予測={dg[:,3].mean():.2f}")

    print(f"\nシード{a.seeds}個 × テスト{a.n_test}件、エポック{a.epochs}、学習{a.n_train}件")
    print(f"{'設定':<14}{'対数RMSE':>22}{'総量比':>7}{'強雨比':>8}{'雨画素RMSE':>11}{'的中率':>7}{'空振り率':>8}")
    for name, ms in rows.items():
        r = np.concatenate([m["rmse"] for m in ms])
        mr, rl, rh = bootstrap_ci(r)
        g = lambda k: np.mean([m[k] for m in ms])
        print(f"{name:<12}{mr:7.3f} [{rl:.3f},{rh:.3f}]{g('total'):+7.2f}{g('strong'):+8.2f}"
              f"{g('wet_rmse'):11.3f}{g('pod'):7.2f}{g('far'):8.2f}")


if __name__ == "__main__":
    main()
