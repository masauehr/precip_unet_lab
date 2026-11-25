"""Phase 5b: ブロック交差検証（学習＋検証の30ブロックを6分割）で、補正モデルが基準より良いかを確かめる。

- ブロックを番号順に並べ、並びの順位 mod 6 で分割（季節が各分割に散らばる）。各分割で 残り5分割を学習・1分割を評価。
- 全分割の評価結果をつなぐと、30ブロック全部が一度ずつ「学習に使っていない状態」で予測される（OOF）。
- テスト集合は使わない。学習シードごとに別プロセスで回せる（結果は outputs/phase5b/ に保存・再開可能）。
使い方: .venv/bin/python src/phase5b.py --seed 0 [--epochs 10] [--folds 6]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5a import day_sets
from real_model import load_set, predict, train
from split import make_split

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase5b"


def build(folds):
    """学習＋検証の日・ブロック・分割番号を返す。"""
    sets, block = make_split()
    days = sets["train"] + sets["val"]
    ub = sorted({block[s] for s in days})
    fold_of_block = {b: i % folds for i, b in enumerate(ub)}
    return days, [block[s] for s in days], [fold_of_block[block[s]] for s in days]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    days, blocks, fold = build(a.folds)
    t0 = time.time()
    x1, y1, b1 = load_set(make_split()[0]["train"], "cache_train.npz")
    x2, y2, b2 = load_set(make_split()[0]["val"], "cache_val.npz")
    x, y, bad = torch.cat([x1, x2]), torch.cat([y1, y2]), torch.cat([b1, b2])
    del x1, y1, b1, x2, y2, b2
    fold_arr = np.repeat(np.array(fold), 24)
    print(f"読み込み {time.time() - t0:.0f}s  {len(days)}日・{len(set(blocks))}ブロック・{a.folds}分割", flush=True)
    for k in range(a.folds):
        f = OUT / f"seed{a.seed}_fold{k}.pkl"
        if f.exists():
            print(f"分割{k}: 取得済み", flush=True)
            continue
        tr, te = np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]
        tdays = [i for i in range(len(days)) if fold[i] == k]
        print(f"=== 分割{k}: 学習 {len(tr) // 24}日 / 評価 {len(te) // 24}日（{len({blocks[i] for i in tdays})}ブロック）seed={a.seed} ===", flush=True)
        model = train(x[tr], y[tr], bad[tr], seed=a.seed, epochs=a.epochs,
                      log=lambda m: print(f"  {m}  経過 {time.time() - t0:.0f}s", flush=True))
        xe, ye, be = x[te], y[te], bad[te]
        pe = predict(model, xe)
        res = {
            "days": [days[i] for i in tdays], "blocks": [blocks[i] for i in tdays],
            "model": day_sets(pe, ye, be, len(tdays)), "base": day_sets(xe, ye, be, len(tdays)),
        }
        np.savez_compressed(OUT / f"pred_seed{a.seed}_fold{k}.npz", pred=pe.numpy().astype(np.float16))
        with open(f, "wb") as fh:
            pickle.dump(res, fh)
        print(f"分割{k} 完了 経過 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
