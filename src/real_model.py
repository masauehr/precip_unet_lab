"""実データ（GSM×GSMaP）の補正モデル: データの束ね・欠測を除く Tweedie 損失・学習・推論。

モデルは合成データで決めた既定（HybridCorrector: 乗法＋加法＋近傍、Tweedie power=1.5、U-Net 深さ3）。
入力はベース場（GSM を GSMaP 格子に載せたもの）80×100。U-Net が深さ3で 8 の倍数を要するため、推論時は右に4列
（端の値の複製）を足して 80×104 にし、出力を元の幅に切り戻す。学習は 64×64 のランダムな切り出し。
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import HybridCorrector, MU_MIN
from real_data import load_day

PAD_W = 4          # 100 → 104（8 の倍数）
CROP = 64


def load_set(days, cache=None):
    """日の一覧 → (ベース場, 真値, 欠測マスク) の torch テンソル。形状 (日×24, 1, 80, 100)。

    cache にファイル名（data/ 配下の .npz）を渡すと、日の一覧が同じ間は再利用する。
    """
    f = Path(__file__).resolve().parents[1] / "data" / cache if cache else None
    if f is not None and f.exists():
        z = np.load(f)
        if list(z["days"]) == list(days):
            return tuple(torch.from_numpy(z[k]) for k in ("gs", "gm", "bad"))
    gs, gm, bad = zip(*(load_day(s) for s in days))
    arr = [np.concatenate(a)[:, None] for a in (gs, gm, bad)]
    if f is not None:
        np.savez(f, days=np.array(days), gs=arr[0], gm=arr[1], bad=arr[2])
    return tuple(torch.from_numpy(a) for a in arr)


def masked_tweedie(mu, target, ok, power=1.5):
    """欠測画素（ok=False）を除いた Tweedie 偏差の平均。"""
    mu = mu.clamp_min(MU_MIN)
    a, b = 2.0 - power, 1.0 - power
    d = 2.0 * (target.pow(a) / (b * a) - target * mu.pow(b) / b + mu.pow(a) / a)
    return (d * ok).sum() / ok.sum().clamp_min(1)


def make_model(base=16, depth=3, **kw):
    return HybridCorrector(base=base, depth=depth, **kw)


def predict(model, x, bs=64):
    """ベース場 (N,1,80,100) → 予測量 (N,1,80,100)。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = F.pad(x[i:i + bs], (0, PAD_W, 0, 0), mode="replicate")
            amount, _ = model.logits(xb)
            out.append(amount[..., :x.shape[-1]])
    return torch.cat(out)


def val_loss(model, x, y, bad, power=1.5):
    p = predict(model, x)
    return float(masked_tweedie(p, y, (~bad).float(), power))


def train(x, y, bad, seed=0, epochs=8, bs=32, lr=2e-3, power=1.5, log=print, val=None, model=None):
    """ランダムな 64×64 の切り出しで学習する。val=(x,y,bad) を渡すと、各エポックの検証損失も表示する。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model or make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n, H, W = x.shape[0], x.shape[2], x.shape[3]
    ok = (~bad).float()
    steps = max(n // bs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(steps):
            ids = perm[s * bs:(s + 1) * bs]
            i0, j0 = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
            sl = (ids, slice(None), slice(i0, i0 + CROP), slice(j0, j0 + CROP))
            amount, _ = model.logits(x[sl])
            loss = masked_tweedie(amount, y[sl], ok[sl], power)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        msg = f"epoch {ep + 1:2d}/{epochs}  学習損失 {tot / steps:.4f}"
        if val is not None:
            msg += f"  検証損失 {val_loss(model, *val, power):.4f}"
        log(msg)
    return model.eval()
