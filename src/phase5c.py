"""Phase 5c: 保存済みの交差検証予測（OOF）に後処理（分位点写像）を掛け、強雨の頻度・CSI が戻るかを見る。

分位点写像（QM）: 予測値の分布（分位点）を、真値の分布へ合わせる単調な写像。各分割 k の評価では、
他の5分割（学習に使っていない状態の予測と真値の組）から写像を求め、分割 k に適用する（k の真値は写像に使わない）。
比べる対象:
  基準 / 基準+QM / 補正モデル / 補正モデル+QM / 補正モデル max(元,QM)（QM で下がる所は元のまま・上がる所だけ上げる）
  ※ 補正モデルの OOF は、他の分割の真値で学習したモデルの予測なので、写像の当てはめに弱い二次的な漏れがある（記録のみ）。
テスト集合は使わない。使い方: .venv/bin/python src/phase5c.py [--seeds 0 1 2]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5a import SHOW, day_sets, paired_ci
from phase5b import OUT as OUT_B, build
from real_baseline import metrics
from real_model import load_set
from split import make_split

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase5c"
LEVELS = np.concatenate([np.linspace(0.0, 0.99, 991), 1 - np.logspace(-2, -6, 241)[1:]])


def fit_qm(pred, true, ok):
    """予測値→真値の分位点写像 (xp, fp)。欠測画素は除く。同じ予測値が続く所は最初の真値分位を採る。"""
    p, t = pred[ok], true[ok]
    xp, fp = np.quantile(p, LEVELS), np.quantile(t, LEVELS)
    xp, idx = np.unique(xp, return_index=True)
    return xp, fp[idx]


def apply_qm(m, a):
    return np.interp(a, m[0], m[1]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days, blocks, fold = build(a.folds)
    fold_arr = np.repeat(np.array(fold), 24)
    sets = make_split()[0]
    x1, y1, b1 = load_set(sets["train"], "cache_train.npz")
    x2, y2, b2 = load_set(sets["val"], "cache_val.npz")
    X, Y, BAD = (torch.cat(p).numpy()[:, 0] for p in ((x1, x2), (y1, y2), (b1, b2)))
    OK = ~BAD
    del x1, y1, b1, x2, y2, b2
    n = len(days)
    t0 = time.time()
    S = {}

    def stats(P):
        return day_sets(torch.from_numpy(P[:, None]), torch.from_numpy(Y[:, None]), torch.from_numpy(BAD[:, None]), n)

    # 基準・基準+QM（シードに依らない）
    Pq = np.zeros_like(X)
    for k in range(a.folds):
        te, fit = fold_arr == k, fold_arr != k
        Pq[te] = apply_qm(fit_qm(X[fit], Y[fit], OK[fit]), X[te])
    S["基準"], S["基準+QM"] = stats(X), stats(Pq)
    print(f"基準 完了 {time.time() - t0:.0f}s", flush=True)

    for sd in a.seeds:
        Pm = np.zeros_like(X)
        for k in range(a.folds):
            Pm[fold_arr == k] = np.load(OUT_B / f"pred_seed{sd}_fold{k}.npz")["pred"][:, 0].astype(np.float32)
        Pqm, Pmx = np.zeros_like(X), np.zeros_like(X)
        for k in range(a.folds):
            te, fit = fold_arr == k, fold_arr != k
            Pqm[te] = apply_qm(fit_qm(Pm[fit], Y[fit], OK[fit]), Pm[te])
        Pmx = np.maximum(Pm, Pqm)
        S[f"補正{sd}"], S[f"補正{sd}+QM"], S[f"補正{sd}max"] = stats(Pm), stats(Pqm), stats(Pmx)
        print(f"seed {sd} 完了 {time.time() - t0:.0f}s", flush=True)
    with open(OUT / "stats.pkl", "wb") as f:
        pickle.dump({"S": S, "blocks": blocks}, f)
    report(S, blocks, a.seeds)


def report(S, blocks, seeds):
    avg = lambda tag: {k: np.mean([metrics(S[f"補正{s}{tag}"])[k] for s in seeds]) for k in SHOW}
    rows = {"基準": metrics(S["基準"]), "基準+QM": metrics(S["基準+QM"]),
            "補正モデル": avg(""), "補正+QM": avg("+QM"), "補正max(元,QM)": avg("max")}
    print(f"\nOOF 297日（補正はシード{seeds}の平均）")
    print(f"{'指標':<10}" + "".join(f"{k:>15}" for k in rows))
    for k in SHOW:
        print(f"{k:<10}" + "".join(f"{r[k]:>15.3f}" for r in rows.values()))
    s0 = seeds[0]
    for name, a_, b_ in [("補正+QM − 基準", f"補正{s0}+QM", "基準"), ("補正+QM − 基準+QM", f"補正{s0}+QM", "基準+QM"),
                         ("補正max − 基準+QM", f"補正{s0}max", "基準+QM"), ("基準+QM − 基準", "基準+QM", "基準")]:
        d = paired_ci(S[a_], S[b_], blocks)
        print(f"\n{name}（seed{s0}・ブロック単位の対の差・* は区間が0を含まない）")
        for k in SHOW:
            m, lo, hi = d[k]
            print(f"  {k:<10}{m:>+9.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if (lo > 0 or hi < 0) else ""))


if __name__ == "__main__":
    main()
