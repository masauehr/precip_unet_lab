"""Phase 5a: 実データで補正モデルを学習し、検証集合で基準（GSMそのまま・総量合わせ）と比べる。

学習は学習集合のみ。検証集合は設定の確認用（4ブロックと少なく不安定）。**テスト集合は読み込まない**。
指標・区間は real_baseline と同じ（日ごとの合計→ブロック単位ブートストラップ）。差は同じブロックの再標本化での対の差。
使い方: .venv/bin/python src/phase5a.py [--epochs 8] [--seed 0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_baseline import day_stats, metrics
from real_model import load_set, predict, train
from split import make_split

SHOW = ["対数RMSE", "総量比", "CSI1", "頻度比1", "POD5", "FAR5", "CSI5", "頻度比5", "FSS5_w1", "FSS5_w9",
        "CSI10", "頻度比10", "FSS10_w9"]


def day_sets(pred, true, bad, n_days):
    """(日×24,1,80,100) の予測・真値・欠測 → 日ごとの合計量のリスト。"""
    p, t, b = pred.numpy()[:, 0], true.numpy()[:, 0], bad.numpy()[:, 0]
    return [day_stats(p[i * 24:(i + 1) * 24], t[i * 24:(i + 1) * 24], b[i * 24:(i + 1) * 24]) for i in range(n_days)]


def paired_ci(SA, SB, blocks, n_boot=500, seed=0):
    """A−B の指標差を、ブロック単位の同じ再標本化で出す。"""
    rng = np.random.default_rng(seed)
    ub = sorted(set(blocks))
    ia = {b: [x for x, bb in zip(SA, blocks) if bb == b] for b in ub}
    ib = {b: [x for x, bb in zip(SB, blocks) if bb == b] for b in ub}
    diffs = {k: [] for k in SHOW}
    for _ in range(n_boot):
        pick = rng.choice(len(ub), size=len(ub))
        ma = metrics([x for i in pick for x in ia[ub[i]]])
        mb = metrics([x for i in pick for x in ib[ub[i]]])
        for k in SHOW:
            diffs[k].append(ma[k] - mb[k])
    return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in diffs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="outputs/phase5a")
    a = ap.parse_args()
    torch.set_num_threads(4)
    sets, block = make_split()
    t0 = time.time()
    xt, yt, bt = load_set(sets["train"], "cache_train.npz")
    xv, yv, bv = load_set(sets["val"], "cache_val.npz")
    print(f"読み込み {time.time() - t0:.0f}s  学習 {len(xt)} 時間・検証 {len(xv)} 時間", flush=True)

    scale = float(yt[~bt].sum() / xt[~bt].sum())
    model = train(xt, yt, bt, seed=a.seed, epochs=a.epochs, val=(xv, yv, bv), log=lambda m: print(f"{m}  経過 {time.time() - t0:.0f}s", flush=True))
    Path(a.save).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{a.save}/model_seed{a.seed}.pt")

    nv = len(sets["val"])
    blocks = [block[s] for s in sets["val"]]
    P = {"補正モデル": predict(model, xv), "GSMそのまま": xv, f"GSM×{scale:.3f}": xv * scale}
    S = {k: day_sets(v, yv, bv, nv) for k, v in P.items()}
    M = {k: metrics(v) for k, v in S.items()}
    print(f"\n検証集合（{nv}日・{len(set(blocks))}ブロック）seed={a.seed}")
    print(f"{'指標':<10}" + "".join(f"{k:>16}" for k in P))
    for k in SHOW:
        print(f"{k:<10}" + "".join(f"{M[n][k]:>16.3f}" for n in P))
    d = paired_ci(S["補正モデル"], S["GSMそのまま"], blocks)
    print("\n補正モデル − GSMそのまま（ブロック単位・対の差。区間が0を含まなければ差は誤差でない）")
    for k in SHOW:
        m, lo, hi = d[k]
        mark = "" if lo <= 0 <= hi else "  *"
        print(f"{k:<10}{m:>+9.3f} [{lo:+.3f}, {hi:+.3f}]{mark}")
    np.savez_compressed(f"{a.save}/val_pred_seed{a.seed}.npz", pred=P["補正モデル"].numpy().astype(np.float16))


if __name__ == "__main__":
    main()
