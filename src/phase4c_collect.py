"""Phase 4c: 確率の差を二段ブートストラップで確かめるための部分和を収集する（ランごとに保存・再開可能）。

設定: 基準の確率 / 単一ネット / アンサンブル（メンバー K=1,2,4,8 を、同じ8メンバーの先頭 K 個で作る）
しきい値 1・5・10 mm/h。部分和は事例ごと（boot.PROB_KEYS）。
あわせて、アンサンブル（K=8）のばらつきと誤差を事例ごとに保存する（降水量を除いた相関の確認用）。
保存先: outputs/phase4c/run_<シード>.npz

使い方: .venv/bin/python src/phase4c_collect.py [--runs 6] [--members 8]   （約1時間）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prob
from model import PROB_THRS
from phase4b import infer, train_prob
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase4c"
KS = (1, 2, 4, 8)


def parts(pt, ptr, o_te, o_tr):
    """事例ごとの部分和。確率しきい値 τ は学習データで CSI 最大のものを使う。"""
    clim = o_tr.mean()
    tau = prob.best_tau(ptr.ravel(), o_tr.ravel())
    pr = pt >= tau
    ax = (1, 2)
    return dict(
        bs=((pt - o_te) ** 2).sum(axis=ax), ref=((clim - o_te) ** 2).sum(axis=ax),
        h=(pr & (o_te > 0)).sum(axis=ax).astype(float),
        fa=(pr & (o_te == 0)).sum(axis=ax).astype(float),
        ms=(~pr & (o_te > 0)).sum(axis=ax).astype(float),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--members", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    T = lambda z: torch.tensor(z[:, None], dtype=torch.float32)
    for r in range(a.runs):
        s = 100 + r
        path = OUT / f"run_{s}.npz"
        if path.exists():
            continue
        t0 = time.time()
        c_tr, f_tr = make_dataset(400, N, FACTOR, seed=s)
        c_te, f_te = make_dataset(200, N, FACTOR, seed=20_000 + s)
        b_tr, b_te = regrid_conservative(c_tr, (N, N)), regrid_conservative(c_te, (N, N))
        x_tr, y_tr, x_te = T(b_tr), T(f_tr), T(b_te)

        single = train_prob(x_tr, y_tr, s)
        mem = [train_prob(x_tr, y_tr, 1000 * (r + 1) + j, boot=True) for j in range(a.members)]
        te = {"単一": infer(single, x_te)}
        tr = {"単一": infer(single, x_tr)[1]}
        mu_te = [infer(m, x_te) for m in mem]
        p_tr_m = [infer(m, x_tr)[1] for m in mem]
        for K in KS:
            te[f"ens{K}"] = (np.mean([o[0] for o in mu_te[:K]], axis=0), np.mean([o[1] for o in mu_te[:K]], axis=0))
            tr[f"ens{K}"] = np.mean(p_tr_m[:K], axis=0)

        save = {}
        for k, t in enumerate(PROB_THRS):
            o_te, o_tr = (f_te >= t).astype(float), (f_tr >= t).astype(float)
            cfgs = {"基準": (prob.binned_baseline(b_tr.ravel(), o_tr.ravel(), b_te.ravel()).reshape(o_te.shape),
                           prob.binned_baseline(b_tr.ravel(), o_tr.ravel(), b_tr.ravel()).reshape(o_tr.shape))}
            for name in te:
                pte = prob.monotone([te[name][1][:, j] for j in range(len(PROB_THRS))])[k]
                ptr = prob.monotone([tr[name][:, j] for j in range(len(PROB_THRS))])[k]
                cfgs[name] = (pte, ptr)
            for name, (pt, ptr) in cfgs.items():
                for f, v in parts(pt, ptr, o_te, o_tr).items():
                    save[f"{name}|{t:g}|{f}"] = v
        # アンサンブル(K=8)のばらつきと誤差（事例ごと）。線形・対数の両方
        mus = np.stack([o[0] for o in mu_te[:8]])
        muE = mus.mean(axis=0)
        save["sp_lin"] = mus.std(axis=0).mean(axis=(1, 2))
        save["sp_log"] = np.log(mus + 0.1).std(axis=0).mean(axis=(1, 2))
        save["err"] = np.sqrt(((np.log(muE + 0.1) - np.log(f_te + 0.1)) ** 2).mean(axis=(1, 2)))
        save["rain"] = f_te.mean(axis=(1, 2))
        # 量の指標（K=8 と単一）
        np.savez(path, **save)
        print(f"ラン {s} 完了 {time.time() - t0:.0f}秒", flush=True)


if __name__ == "__main__":
    main()
