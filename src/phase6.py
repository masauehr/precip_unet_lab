"""Phase 6（探索）: 予報時間（FT・時刻）を入力に加えると、補正が良くなるか。ブロック交差検証（Phase 5b と同じ分割・3シード）。

GSM は 00UTC 初期値で、FT=k の降水は k−1〜k 時（UTC）。**時刻（日変化）**の情報として、sin/cos(2π·FT/24) の2チャンネルを、
ベース場の隣に入力する（空間一様の定数マップ）。他は Phase 5b と同じ（HybridCorrector・Tweedie 1.5・10エポック）。
テスト集合・新しい期間は使わない（Phase 5b の OOF と比べるだけの探索。採用するなら、未使用の新しい期間で確かめる必要がある）。
使い方: .venv/bin/python src/phase6.py --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EPS, HybridCorrector
from phase5b import build
from real_model import CROP, PAD_W, load_set, masked_tweedie
from split import make_split

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase6"


class HybridFT(HybridCorrector):
    """入力 x は (B, 3, H, W) = [ベース場, sin(FT), cos(FT)]。"""

    def __init__(self, base=16, depth=3, n_extra=2, **kw):
        super().__init__(base=base, depth=depth, **kw)
        from model import UNet
        self.unet = UNet(c_in=3 + n_extra, base=base, depth=depth, c_out=3)   # log(ベース) + 近傍2 + 追加チャンネル
        with torch.no_grad():
            self.unet.head.bias[2] = -6.0

    def logits(self, x):
        base, tf = x[:, :1], x[:, 1:]
        out = self.unet(torch.cat([self._features(base), tf], dim=1))
        amount = base * torch.exp(out[:, :1]) + F.softplus(out[:, 2:3])
        return amount, out[:, 1:2]


def add_ft(x):
    """(N,1,H,W)（N=日×24・時刻順）→ (N,3,H,W)。FT=(index%24)+1。"""
    n = x.shape[0]
    ft = (torch.arange(n) % 24 + 1).float()
    s, c = torch.sin(2 * np.pi * ft / 24), torch.cos(2 * np.pi * ft / 24)
    f = lambda v: v.view(n, 1, 1, 1).expand(n, 1, x.shape[2], x.shape[3])
    return torch.cat([x, f(s), f(c)], dim=1)


def predict(model, x, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = F.pad(x[i:i + bs], (0, PAD_W, 0, 0), mode="replicate")
            amount, _ = model.logits(xb)
            out.append(amount[..., :x.shape[-1]])
    return torch.cat(out)


def train(x, y, bad, seed, epochs=10, bs=32, lr=2e-3, log=print, model_fn=HybridFT):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_fn()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n, H, W = x.shape[0], x.shape[2], x.shape[3]
    ok = (~bad).float()
    steps = max(n // bs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(n), 0.0
        for s in range(steps):
            ids = perm[s * bs:(s + 1) * bs]
            i0, j0 = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
            sl = (ids, slice(None), slice(i0, i0 + CROP), slice(j0, j0 + CROP))
            amount, _ = model.logits(x[sl])
            loss = masked_tweedie(amount, y[sl], ok[sl])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach())
        log(f"epoch {ep + 1:2d}/{epochs}  学習損失 {tot / steps:.4f}")
    return model.eval()


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
    x = add_ft(x)
    t0 = time.time()
    for k in range(6):
        f = OUT / f"pred_seed{a.seed}_fold{k}.npz"
        if f.exists():
            continue
        tr, te = np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]
        print(f"=== 分割{k} seed={a.seed} 学習{len(tr) // 24}日 / 評価{len(te) // 24}日 ===", flush=True)
        model = train(x[tr], y[tr], bad[tr], a.seed, a.epochs, log=lambda m: print(f"  {m}  経過 {time.time() - t0:.0f}s", flush=True))
        p = predict(model, x[te])
        np.savez_compressed(f, pred=p.numpy().astype(np.float16))
        print(f"分割{k} 完了 経過 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
