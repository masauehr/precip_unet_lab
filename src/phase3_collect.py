"""Phase 3: 設定ごと・シードごとに学習し、事例ごとの部分和を保存する（再開可能）。

Phase 2 の選択に使っていない新しいシード（学習 100〜、テスト 20000+）で、
  既定 / power1.2 / 最大プーリング / 加法なし(参考: 差が大きいはずの対照) を比べる材料を作る。
保存先: outputs/phase3/<設定>_s<シード>.npz （boot.py の KEYS）

使い方: .venv/bin/python src/phase3_collect.py [--seeds 8]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import WET
from phase2 import VARIANTS, train
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N

SEL = ["既定", "power1.2", "最大プーリング", "加法なし(乗法のみ)"]
OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase3"


def case_parts(pred, true, thr=5.0, win=9):
    """事例ごとの部分和（boot.KEYS）。"""
    d2 = (np.log(pred + 0.1) - np.log(true + 0.1)) ** 2
    wet, strong = true >= WET, true >= thr
    fr = lambda a: F.avg_pool2d(torch.tensor((a >= thr).astype(np.float32))[:, None], win,
                                stride=1, padding=win // 2, count_include_pad=False).numpy()[:, 0]
    pf, of = fr(pred), fr(true)
    ax = (1, 2)
    return dict(
        rmse=np.sqrt(d2.mean(axis=ax)),
        sp=pred.sum(axis=ax), st=true.sum(axis=ax),
        sps=(pred * strong).sum(axis=ax), sts=(true * strong).sum(axis=ax),
        wsq=(d2 * wet).sum(axis=ax), wn=wet.sum(axis=ax).astype(float),
        fnum=((pf - of) ** 2).sum(axis=ax), fden=(pf**2 + of**2).sum(axis=ax),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {n: (kw, p) for n, kw, p in VARIANTS}
    for k in range(a.seeds):
        s = 100 + k
        c, f = make_dataset(a.n_test, N, FACTOR, seed=20_000 + s)
        x = torch.tensor(regrid_conservative(c, (N, N))[:, None], dtype=torch.float32)
        for name in SEL:
            path = OUT / f"{name}_s{s}.npz"
            if path.exists():
                continue
            t0 = time.time()
            kw, power = cfg[name]
            m, _ = train(kw, power, s, a.epochs, a.n_train)
            with torch.no_grad():
                pred = m(x)[0].numpy()[:, 0]
            np.savez(path, **case_parts(pred, f))
            print(f"{name} シード{s} 完了 {time.time() - t0:.0f}秒", flush=True)


if __name__ == "__main__":
    main()
