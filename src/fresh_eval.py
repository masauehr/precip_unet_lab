"""新しい期間（2026-05-11〜）での再評価。最終評価（テスト）とは別の、**学習・検証・テストのどれにも使っていない日**。

固定した最終モデル・QM・τ（outputs/final/）をそのまま使い、設定は変えない。
GSMaP は 2026-05-11 以降、各日の UTC 0・1 時が配信されない（-99 で欠測扱い）。データは data/gsmap_new/（分割を凍結した本体と分ける）。
先に時刻整合（lag 相関）を確認してから評価する。ブロックは日付順の10日ごと。
使い方: .venv/bin/python src/fresh_eval.py [--from YYYYMMDD]
  --from を付けると、その日以降だけを評価する（まだ評価に使っていない日だけを見る）。新しい GSMaP の再評価手順はマニュアル「新しいGSMaPでの再評価の手順」。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import report
from real_data import gsm_on_gsmap

D = Path(__file__).resolve().parents[1] / "data"


def load_fresh(from_day=None):
    days = []
    for p in sorted((D / "gsmap_new").glob("2*.npy")):
        if from_day and p.stem < from_day:
            continue
        if (D / "gsm" / p.name).exists() and not (np.load(p) < -90).all():
            days.append(p.stem)
    gs, gm, bad = [], [], []
    for s in days:
        g = np.load(D / "gsmap_new" / f"{s}.npy")
        b = ~np.isfinite(g) | (g < 0)
        gs.append(np.where(b, 0.0, gsm_on_gsmap(np.load(D / "gsm" / f"{s}.npy"))).astype(np.float32))
        gm.append(np.where(b, 0.0, g).astype(np.float32))
        bad.append(b)
    t = lambda a: torch.from_numpy(np.concatenate(a)[:, None])
    return days, t(gs), t(gm), t(bad)


def lag_check(x, y, bad):
    """GSM と GSMaP の領域平均（有効な時刻のみ）の相関を、ずらし幅ごとに見る。ピークが 0 なら整合。"""
    X, Y, B = x.numpy()[:, 0], y.numpy()[:, 0], bad.numpy()[:, 0]
    nd = len(X) // 24
    ok = (~B).reshape(nd, 24, -1).mean(axis=2) > 0.5
    mx, my = X.reshape(nd, 24, -1).mean(axis=2), Y.reshape(nd, 24, -1).mean(axis=2)
    print("時刻整合（領域平均の相関・ずらし幅 h）")
    for lag in (-2, -1, 0, 1, 2):
        a, b, m = [], [], []
        for h in range(24):
            h2 = h + lag
            if 0 <= h2 < 24:
                a.append(mx[:, h]); b.append(my[:, h2]); m.append(ok[:, h] & ok[:, h2])
        a, b, m = np.concatenate(a), np.concatenate(b), np.concatenate(m)
        print(f"  lag {lag:+d}: {np.corrcoef(a[m], b[m])[0, 1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_day", default=None, help="この日（YYYYMMDD）以降だけを評価する")
    a = ap.parse_args()
    days, x, y, bad = load_fresh(a.from_day)
    blocks = [i // 10 for i in range(len(days))]
    print(f"新しい期間: {len(days)}日（{days[0]}〜{days[-1]}）・{len(set(blocks))}ブロック")
    lag_check(x, y, bad)
    report(x, y, bad, blocks, "fresh")


if __name__ == "__main__":
    main()
