"""最終評価（テスト集合・**1回だけ**）。方針と主たる問いは FINAL_PROTOCOL.md。

事前に src/final_fit.py で最終モデル（outputs/final/*.pt）と較正（calib.pkl）を作っておく。
実行すると outputs/final/EVALUATED を作り、2回目以降は拒否する（結果を見て設定を変えないため）。
使い方: .venv/bin/python src/final_eval.py [--dry]   （--dry は検証集合で配管だけ確かめる。印を作らない）
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PROB_THRS
from final_fit import OUT, SEEDS
from phase5a import SHOW, day_sets, paired_ci
from phase5c import apply_qm
from phase5d import predict as predict_prob
from phase5d_report import boot, counts
from prob import binned_baseline, ece, reliability
from real_baseline import block_ci, metrics
from real_model import load_set, make_model, predict
from model import ProbCorrector
from split import make_split


def mean_stats(list_of_S):
    """シードごとの日別集計を平均する（件数・和は加法的なので、平均してから指標にする）。"""
    return [{k: float(np.mean([S[i][k] for S in list_of_S])) for k in list_of_S[0][0]} for i in range(len(list_of_S[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    mark = OUT / "EVALUATED"
    if not a.dry and mark.exists():
        sys.exit("最終評価は実施済みです（outputs/final/EVALUATED）。設定を変えて再評価しないでください。")
    sets, block = make_split()
    which = "val" if a.dry else "test"
    days = sets[which]
    blocks = [block[s] for s in days]
    n = len(days)
    x, y, bad = load_set(days, f"cache_{which}.npz")
    X, Y, BAD = x.numpy()[:, 0], y.numpy()[:, 0], bad.numpy()[:, 0]
    OK = ~BAD
    C = pickle.load(open(OUT / "calib.pkl", "rb"))
    stats = lambda P: day_sets(torch.from_numpy(P[:, None]), y, bad, n)
    R = {"基準": stats(X), "基準+QM": stats(apply_qm(C["qm_base"], X))}
    per = []
    for s in SEEDS:
        m = make_model()
        m.load_state_dict(torch.load(OUT / f"hybrid_seed{s}.pt"))
        P = predict(m.eval(), x).numpy()[:, 0]
        per.append(stats(apply_qm(C["qm_model"][s], P)))
    R["補正+QM"] = mean_stats(per)
    print(f"■ 集合 {which}: {n}日・{len(set(blocks))}ブロック")
    ci = block_ci(R["基準"], blocks)
    rows = {k: metrics(v) for k, v in R.items()}
    print(f"{'指標':<10}" + "".join(f"{k:>14}" for k in rows))
    for k in SHOW:
        print(f"{k:<10}" + "".join(f"{r[k]:>14.3f}" for r in rows.values()))
    print("\n【主たる問い1】補正+QM − 基準+QM（ブロック単位・対の差・* は区間が0を含まない）")
    d = paired_ci(R["補正+QM"], R["基準+QM"], blocks)
    for k in SHOW:
        mm, lo, hi = d[k]
        main_ = "◎" if k in ("対数RMSE", "CSI1", "CSI5", "FAR5") else " "
        print(f" {main_} {k:<10}{mm:>+9.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if (lo > 0 or hi < 0) else ""))
    print("\n参考: 補正+QM − 基準（QMなし）")
    d = paired_ci(R["補正+QM"], R["基準"], blocks)
    for k in SHOW:
        mm, lo, hi = d[k]
        print(f"    {k:<10}{mm:>+9.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if (lo > 0 or hi < 0) else ""))

    # ---- 確率
    pm = []
    for s in SEEDS:
        m = ProbCorrector(base=16, depth=3)
        m.load_state_dict(torch.load(OUT / f"prob_seed{s}.pt"))
        pm.append(predict_prob(m.eval(), x)[1].numpy())
    ens = np.mean(pm, axis=0)                     # (N,3,80,100)
    print("\n【主たる問い2】確率: BSS（気候値に対して）・階級別基準との差")
    res = {}
    for j, t in enumerate(PROB_THRS):
        c = C["prob"][t]
        o = (Y >= t)
        bb = binned_baseline(c["base_x"], c["base_o"], X.ravel()).reshape(X.shape)
        p = ens[:, j]
        r = lambda q: ((q - o) ** 2 * OK).reshape(n, -1).sum(axis=1)
        sse = {"気候値": r(np.full_like(bb, c["clim"])), "階級別基準": r(bb), "アンサンブル": r(p)}
        bss = {m: 1 - sse[m].sum() / sse["気候値"].sum() for m in ("階級別基準", "アンサンブル")}
        f = lambda sa, sb, sc: (1 - sa / sc) - (1 - sb / sc)
        mm, lo, hi = boot(f, blocks, [sse["アンサンブル"], sse["階級別基準"], sse["気候値"]])
        e = ece(p[OK][::7], o[OK][::7].astype(float))
        print(f"≥{t:>4g}mm/h 発生率 {o[OK].mean():.4f}  BSS 階級別基準 {bss['階級別基準']:.3f} アンサンブル {bss['アンサンブル']:.3f}"
              f"  差 {mm:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if (lo > 0 or hi < 0) else "") + f"  ECE {e:.4f}")
        print("   信頼性: " + "  ".join(f"[{pp:.2f}/{oo:.2f}/{cc}]" for pp, oo, cc in reliability(p[OK][::7], o[OK][::7].astype(float), 10) if cc > 0))
        res[t] = (counts(p >= c["tau_ens"], o, OK), counts(bb >= c["tau_base"], o, OK))
    print("\n参考: 確率を固定 τ で二値化した CSI / POD / FAR / 頻度比（補正+QM は同じ指標を日別集計から）")
    for t in PROB_THRS:
        c_ens, c_base = res[t]
        c_qm = np.array([[x_[f"h{t}"], x_[f"fa{t}"], x_[f"ms{t}"]] for x_ in R["補正+QM"]])
        print(f"≥{t:g}mm/h  τ(アンサンブル)={C['prob'][t]['tau_ens']:.2f}")
        for nm, cc in (("階級別基準", c_base), ("アンサンブル", c_ens), ("補正+QM", c_qm)):
            fs = (lambda h, fa, ms: h / max(h + fa + ms, 1), lambda h, fa, ms: h / max(h + ms, 1),
                  lambda h, fa, ms: fa / max(h + fa, 1), lambda h, fa, ms: (h + fa) / max(h + ms, 1))
            v = [boot(f, blocks, [cc[:, 0], cc[:, 1], cc[:, 2]]) for f in fs]
            print(f"  {nm:<12}" + "  ".join(f"{lab} {u[0]:.3f}[{u[1]:.3f},{u[2]:.3f}]" for lab, u in zip(("CSI", "POD", "FAR", "頻度比"), v)))
    if not a.dry:
        mark.write_text("最終評価を実施済み。設定を変えて再評価しない。\n")


if __name__ == "__main__":
    main()
