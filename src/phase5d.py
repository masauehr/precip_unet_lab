"""Phase 5d: 超過確率モデル（ProbCorrector: 量＋しきい値 1・5・10 mm/h の超過確率）のブロック交差検証。

Phase 5b と同じ分割（30ブロック・6分割）。損失は 欠測を除いた Tweedie（量）＋ BCE（各しきい値の確率）の平均。
OOF（学習に使っていない状態）の量と確率を outputs/phase5d/ に保存。テスト集合は使わない。
使い方: .venv/bin/python src/phase5d.py --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PROB_THRS, ProbCorrector
from phase5b import build
from real_model import CROP, PAD_W, load_set, masked_tweedie
from split import make_split

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase5d"


def masked_prob_loss(mu, logits, target, ok, power=1.5):
    bce = sum((F.binary_cross_entropy_with_logits(logits[:, k:k + 1], (target >= t).float(), reduction="none") * ok).sum()
              / ok.sum().clamp_min(1) for k, t in enumerate(PROB_THRS)) / len(PROB_THRS)
    return masked_tweedie(mu, target, ok, power) + bce


def train(x, y, bad, seed, epochs=10, bs=32, lr=2e-3, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ProbCorrector(base=16, depth=3)
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
            mu, z = model.heads(x[sl])
            loss = masked_prob_loss(mu, z, y[sl], ok[sl])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach())
        log(f"epoch {ep + 1:2d}/{epochs}  学習損失 {tot / steps:.4f}")
    return model.eval()


def predict(model, x, bs=64):
    """ベース場 → (量, 確率3チャンネル)。"""
    mus, ps = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = F.pad(x[i:i + bs], (0, PAD_W, 0, 0), mode="replicate")
            mu, z = model.heads(xb)
            mus.append(mu[..., :x.shape[-1]]); ps.append(torch.sigmoid(z)[..., :x.shape[-1]])
    return torch.cat(mus), torch.cat(ps)


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
    fold_arr = np.repeat(np.array(fold), 24)
    sets = make_split()[0]
    x1, y1, b1 = load_set(sets["train"], "cache_train.npz")
    x2, y2, b2 = load_set(sets["val"], "cache_val.npz")
    x, y, bad = torch.cat([x1, x2]), torch.cat([y1, y2]), torch.cat([b1, b2])
    del x1, y1, b1, x2, y2, b2
    t0 = time.time()
    for k in range(a.folds):
        f = OUT / f"seed{a.seed}_fold{k}.npz"
        if f.exists():
            continue
        tr, te = np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]
        print(f"=== 分割{k} seed={a.seed} 学習{len(tr) // 24}日 / 評価{len(te) // 24}日 ===", flush=True)
        model = train(x[tr], y[tr], bad[tr], a.seed, a.epochs, log=lambda m: print(f"  {m}  経過 {time.time() - t0:.0f}s", flush=True))
        mu, p = predict(model, x[te])
        np.savez_compressed(f, mu=mu.numpy().astype(np.float16), p=p.numpy().astype(np.float16))
        print(f"分割{k} 完了 経過 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
