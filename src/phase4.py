"""Phase 4: 評価指標を広げて、強雨が戻らない理由を切り分ける。

比べる: 再格子化 / 既定(乗法＋加法＋近傍・Tweedie1.5) / 既定＋分位点写像 / 加法なし。
指標: しきい値別（1・5・10 mm/h）の POD・適合率・CSI・頻度バイアス、窓を変えた FSS、
      強雨の比（真値で条件付け／予測で条件付け）、空間スペクトルの 予測/真値 比。
学習シード 100〜104（Phase 3 と同じ。選択には使っていない）、各テスト200件。

使い方: .venv/bin/python src/phase4.py [--seeds 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_ext import categorical, cond_ratios, fss_window, spectrum_ratio
from phase1c import apply_quantile_map, fit_quantile_map
from phase2 import VARIANTS, train
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N

CFG = {n: (kw, p) for n, kw, p in VARIANTS}
ORDER = ["再格子化", "既定", "既定+分位点", "加法なし"]
THRS = (1.0, 5.0, 10.0)


def infer(m, base):
    with torch.no_grad():
        return m(torch.tensor(base[:, None], dtype=torch.float32))[0].numpy()[:, 0]


def evaluate(pred, true):
    r = {}
    for t in THRS:
        for k, v in categorical(pred, true, t).items():
            r[f"{k}{t:g}"] = v
        a, b = cond_ratios(pred, true, t)
        r[f"cT{t:g}"], r[f"cP{t:g}"] = a, b
    for w in (1, 5, 9, 17):
        r[f"fss5_w{w}"] = fss_window(pred, true, 5.0, w)
    r["spec"] = spectrum_ratio(pred, true)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(4)
    res = {n: [] for n in ORDER}
    for k in range(a.seeds):
        s = 100 + k
        c_te, f_te = make_dataset(200, N, FACTOR, seed=20_000 + s)
        b_te = regrid_conservative(c_te, (N, N))
        c_tr, f_tr = make_dataset(400, N, FACTOR, seed=s)
        b_tr = regrid_conservative(c_tr, (N, N))
        res["再格子化"].append(evaluate(b_te, f_te))
        m, _ = train(*CFG["既定"], s, 15, 400)
        p_te = infer(m, b_te)
        res["既定"].append(evaluate(p_te, f_te))
        qp, qt = fit_quantile_map(infer(m, b_tr), f_tr)
        res["既定+分位点"].append(evaluate(apply_quantile_map(p_te, qp, qt), f_te))
        m2, _ = train(*CFG["加法なし(乗法のみ)"], s, 15, 400)
        res["加法なし"].append(evaluate(infer(m2, b_te), f_te))
        print(f"シード {s} 完了", flush=True)

    mean = lambda n, key: np.nanmean([r[key] for r in res[n]])
    sd = lambda n, key: np.nanstd([r[key] for r in res[n]])
    print(f"\n学習シード{a.seeds}個の平均（テスト各200件）")
    for t in THRS:
        print(f"\n[しきい値 {t:g} mm/h 以上を事象とする]")
        print(f"{'設定':<14}{'POD':>7}{'適合率':>8}{'CSI':>7}{'CSI sd':>8}{'頻度バイアス':>11}")
        for n in ORDER:
            print(f"{n:<12}{mean(n,f'pod{t:g}'):7.3f}{mean(n,f'prec{t:g}'):8.3f}{mean(n,f'csi{t:g}'):7.3f}"
                  f"{sd(n,f'csi{t:g}'):8.3f}{mean(n,f'fbias{t:g}'):11.2f}")
    print("\n[5 mm/h の FSS（窓＝画素数。1は画素どうし、大きいほど位置のずれに寛容）]")
    print(f"{'設定':<14}" + "".join(f"{'窓'+str(w):>8}" for w in (1, 5, 9, 17)))
    for n in ORDER:
        print(f"{n:<12}" + "".join(f"{mean(n,f'fss5_w{w}'):8.3f}" for w in (1, 5, 9, 17)))
    print("\n[強雨の比(log)  A:真値で条件付け=予測平均/真値平均   B:予測で条件付け=真値平均/予測平均（0が理想）]")
    print(f"{'設定':<14}" + "".join(f"{'A≥'+f'{t:g}':>8}{'B≥'+f'{t:g}':>8}" for t in THRS))
    for n in ORDER:
        print(f"{n:<12}" + "".join(f"{mean(n,f'cT{t:g}'):+8.2f}{mean(n,f'cP{t:g}'):+8.2f}" for t in THRS))
    print("\n[空間スペクトルの 予測/真値（1なら細かさが保たれる。波数＝周期/領域、粗い予報の限界は波数4）]")
    print(f"{'設定':<14}" + "".join(f"{b:>10}" for b in ("1〜4", "5〜8", "9〜16", "17〜32")))
    for n in ORDER:
        sp = np.mean([r["spec"] for r in res[n]], axis=0)
        print(f"{n:<12}" + "".join(f"{v:10.2f}" for v in sp))


if __name__ == "__main__":
    main()
