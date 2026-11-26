"""Phase 7（探索）: 前後の時刻の GSM 降水（時間の文脈）を入力に加えると、補正が良くなるか。ブロック交差検証（Phase 5b・6 と同じ分割・3シード）。

GSM は 1 回の予報（00UTC 初期値）の FT=1〜24 が手に入るので、その時刻の前後（FT−1・FT+1）の降水場は、予報時点で使える情報。
雨雲が動く・位置や時刻が少しずれる、という誤差を拾えるかを見る。日の端（FT=1 の前・FT=24 の後）は自分自身で埋める。
入力は [ベース場, 前の時刻, 後の時刻]（ベース場を対数にして、近傍の特徴と合わせて U-Net へ）。テスト・新しい期間は使わない。
使い方: .venv/bin/python src/phase7.py --seed 0
"""
import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6
from model import EPS
from phase5b import build
from real_model import load_set
from split import make_split

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase7"


class HybridCtx(phase6.HybridFT):
    """入力 x は (B, 3, H, W) = [ベース場, 前の時刻, 後の時刻]。前後は対数にして、特徴に加える。"""

    def __init__(self, base=16, depth=3, **kw):
        super().__init__(base=base, depth=depth, n_extra=2, **kw)

    def logits(self, x):
        b = x[:, :1]
        out = self.unet(torch.cat([self._features(b), torch.log(x[:, 1:] + EPS)], dim=1))
        amount = b * torch.exp(out[:, :1]) + torch.nn.functional.softplus(out[:, 2:3])
        return amount, out[:, 1:2]


def add_ctx(x):
    """(N,1,H,W)（N=日×24・時刻順）→ (N,3,H,W)。同じ日の前後の時刻（端は自分自身）。"""
    n = x.shape[0]
    idx = torch.arange(n)
    hr = idx % 24
    prev = torch.where(hr == 0, idx, idx - 1)
    nxt = torch.where(hr == 23, idx, idx + 1)
    return torch.cat([x, x[prev], x[nxt]], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    days, blocks, fold = build(6)
    fold_arr = np.repeat(np.array(fold), 24)
    sets = make_split()[0]
    x1, y1, b1 = load_set(sets["train"], "cache_train.npz")
    x2, y2, b2 = load_set(sets["val"], "cache_val.npz")
    x, y, bad = torch.cat([x1, x2]), torch.cat([y1, y2]), torch.cat([b1, b2])
    x = add_ctx(x)
    t0 = time.time()
    for k in range(6):
        f = OUT / f"pred_seed{a.seed}_fold{k}.npz"
        if f.exists():
            continue
        tr, te = np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]
        print(f"=== 分割{k} seed={a.seed} 学習{len(tr) // 24}日 / 評価{len(te) // 24}日 ===", flush=True)
        model = phase6.train(x[tr], y[tr], bad[tr], a.seed, a.epochs, model_fn=HybridCtx,
                             log=lambda m: print(f"  {m}  経過 {time.time() - t0:.0f}s", flush=True))
        p = phase6.predict(model, x[te])
        np.savez_compressed(f, pred=p.numpy().astype(np.float16))
        print(f"分割{k} 完了 経過 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
