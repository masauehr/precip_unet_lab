"""Phase 5d の集計: 超過確率（1・5・10 mm/h）の精度を OOF で評価する。

比べる確率: 気候値（他分割の発生率）／階級別基準（GSM の強さの階級ごとの発生率）／単一ネット（seed0）／アンサンブル（全シード平均）
指標: ブライアススキル（BSS・気候値に対して）、ECE、信頼性の表。事象は「その時刻の GSMaP が しきい値以上」。
確率→二値は、他分割の OOF で CSI が最大になる確率しきい値 τ を選ぶ。CSI/POD/FAR を Phase5c の QM と並べる。
差は ブロック単位の再標本化（500回）の対の差。テスト集合は使わない。
使い方: .venv/bin/python src/phase5d_report.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PROB_THRS
from phase5b import build
from phase5c import OUT as OUT_C
from phase5d import OUT
from prob import best_tau, binned_baseline, ece, monotone, reliability
from real_model import load_set
from split import make_split

SUB = 4_000_000   # 当てはめ用に抽出する画素数
GRID = np.linspace(0.05, 0.95, 19)


def counts(pred_bin, o, ok):
    """日ごとの 的中・空振り・見逃し の数 (n_days,3)。"""
    nd = o.shape[0] // 24
    r = lambda a: a.reshape(nd, -1).sum(axis=1)
    p, ob = pred_bin & ok, o & ok
    return np.stack([r(p & ob), r(p & ~ob), r(~p & ob)], axis=1).astype(float)


def boot(fn, blocks, arrs, n_boot=500, seed=0):
    """ブロック単位の再標本化で fn(*日ごとの配列の合計) の 95% 区間。"""
    rng = np.random.default_rng(seed)
    bl = np.array(blocks)
    ub = sorted(set(blocks))
    idx = {b: np.where(bl == b)[0] for b in ub}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(ub), size=len(ub))
        ii = np.concatenate([idx[ub[i]] for i in pick])
        out.append(fn(*[a[ii].sum(axis=0) for a in arrs]))
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    days, blocks, fold = build(6)
    fold_arr = np.repeat(np.array(fold), 24)
    seeds = sorted({int(p.name.split("_")[0][4:]) for p in OUT.glob("seed*_fold*.npz")})
    done = [k for k in range(6) if all((OUT / f"seed{s}_fold{k}.npz").exists() for s in seeds)]
    sets = make_split()[0]
    x1, y1, b1 = load_set(sets["train"], "cache_train.npz")
    x2, y2, b2 = load_set(sets["val"], "cache_val.npz")
    X, Y, BAD = (torch.cat(p).numpy()[:, 0] for p in ((x1, x2), (y1, y2), (b1, b2)))
    OK = ~BAD
    di = [i for i in range(len(days)) if fold[i] in done]          # 評価に使う日（完了した分割）
    sel = np.concatenate([np.arange(i * 24, (i + 1) * 24) for i in di])
    n = len(di)
    blk = [blocks[i] for i in di]
    print(f"シード {seeds}・分割 {done}・{n}日・{len(set(blk))}ブロック")
    # OOF の確率を全体の並びに置く
    P = {s: np.zeros((len(X), 3, 80, 100), np.float16) for s in seeds}
    for s in seeds:
        for k in done:
            P[s][fold_arr == k] = np.load(OUT / f"seed{s}_fold{k}.npz")["p"][:, :, :, :]
    rng = np.random.default_rng(0)
    res = {}
    names = ["気候値", "階級別基準", "単一(seed0)", "アンサンブル"]
    for j, t in enumerate(PROB_THRS):
        sse = {m: np.zeros(n) for m in names}
        cnt = {m: np.zeros((n, 3)) for m in ("階級別基準", "単一(seed0)", "アンサンブル")}
        nvalid = np.zeros(n)
        ens_all, o_all, ok_all = [], [], []
        for k in done:
            te = np.where(fold_arr == k)[0]
            fit = np.where(np.isin(fold_arr, [q for q in done if q != k]))[0]
            dd = np.where(np.array(fold)[di] == k)[0]            # この分割の日の、評価用の並び内の位置
            ok_f, o_f = OK[fit], (Y[fit] >= t)
            fx = X[fit][ok_f]; fo = o_f[ok_f].astype(np.float32)
            sub = rng.choice(len(fx), size=min(SUB, len(fx)), replace=False)
            clim = float(fo.mean())
            te_ok, te_o = OK[te], (Y[te] >= t)
            xt = X[te]
            bb = binned_baseline(fx[sub], fo[sub], xt.ravel()).reshape(xt.shape)
            ps = {s: P[s][te][:, j].astype(np.float32) for s in seeds}
            ens = np.mean([P[s][te][:, j].astype(np.float32) for s in seeds], axis=0)
            # 単調化（しきい値が上がるほど確率が下がる）は、3つ全部を使うので複数チャンネルが要る→ここでは省略（記録のみ）
            fp_ens = np.mean([P[s][fit][:, j].astype(np.float32) for s in seeds], axis=0)[ok_f]
            fp_one = P[seeds[0]][fit][:, j].astype(np.float32)[ok_f]
            tau_e = best_tau(fp_ens[sub], fo[sub], GRID)
            tau_1 = best_tau(fp_one[sub], fo[sub], GRID)
            tau_b = best_tau(binned_baseline(fx[sub], fo[sub], fx[sub]), fo[sub], GRID)
            nd = len(te) // 24
            r = lambda a: ((a - te_o) ** 2 * te_ok).reshape(nd, -1).sum(axis=1)
            sl = dd
            sse["気候値"][sl] = r(np.full_like(bb, clim))
            sse["階級別基準"][sl], sse["単一(seed0)"][sl], sse["アンサンブル"][sl] = r(bb), r(ps[seeds[0]]), r(ens)
            nvalid[sl] = te_ok.reshape(nd, -1).sum(axis=1)
            cnt["階級別基準"][sl] = counts(bb >= tau_b, te_o, te_ok)
            cnt["単一(seed0)"][sl] = counts(ps[seeds[0]] >= tau_1, te_o, te_ok)
            cnt["アンサンブル"][sl] = counts(ens >= tau_e, te_o, te_ok)
            ens_all.append(ens[te_ok][::7]); o_all.append(te_o[te_ok][::7])
            res.setdefault(("tau", t), []).append((tau_b, tau_1, tau_e))
        res[("sse", t)], res[("cnt", t)], res[("nv", t)] = sse, cnt, nvalid
        res[("rel", t)] = (np.concatenate(ens_all), np.concatenate(o_all).astype(float))
    # ---- 出力
    print("\n■ ブライアススキル（BSS・気候値に対して。大きいほど良い）と ECE（アンサンブル・小さいほど良い）")
    print(f"{'しきい値':<9}{'発生率':>8}" + "".join(f"{m:>14}" for m in names[1:]) + f"{'ECE':>8}")
    for t in PROB_THRS:
        sse = res[("sse", t)]
        bss = {m: 1 - sse[m].sum() / sse["気候値"].sum() for m in names[1:]}
        pe, oe = res[("rel", t)]
        print(f"≥{t:>4g}mm/h{oe.mean():>8.4f}" + "".join(f"{bss[m]:>14.3f}" for m in names[1:]) + f"{ece(pe, oe):>8.4f}")
    print("\n■ BSS の差（ブロック単位の対の差・* は区間が0を含まない）")
    for t in PROB_THRS:
        sse = res[("sse", t)]
        for a, b in [("アンサンブル", "階級別基準"), ("アンサンブル", "単一(seed0)"), ("単一(seed0)", "階級別基準")]:
            f = lambda sa, sb, sc: (1 - sa / sc) - (1 - sb / sc)
            m, lo, hi = boot(f, blk, [sse[a], sse[b], sse["気候値"]])
            print(f"  ≥{t:>4g} {a}−{b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if (lo > 0 or hi < 0) else ""))
    print("\n■ 信頼性（アンサンブル・予測確率の帯ごと: 予測平均 / 観測頻度 / 件数(間引き後)）")
    for t in PROB_THRS:
        pe, oe = res[("rel", t)]
        print(f"≥{t:g}mm/h: " + "  ".join(f"[{p:.2f}/{o:.2f}/{c}]" for p, o, c in reliability(pe, oe, 10) if c > 0))
    print("\n■ 確率をしきい値 τ で二値化した CSI / POD / FAR / 頻度比（τ は他分割の OOF で CSI 最大）と、Phase5c の QM")
    S5c = pickle.load(open(OUT_C / "stats.pkl", "rb"))["S"]
    sel_days = np.array(di)
    for t in PROB_THRS:
        cnt = res[("cnt", t)]
        taus = np.mean(res[("tau", t)], axis=0)
        print(f"≥{t:g}mm/h  τ(階級別/単一/アンサンブル)≒{taus[0]:.2f}/{taus[1]:.2f}/{taus[2]:.2f}")
        rows = []
        for m, c in cnt.items():
            rows.append((m, c))
        for nm, key in [("基準+QM", "基準+QM"), ("補正+QM", f"補正{seeds[0]}+QM")]:
            c = np.array([[S5c[key][i][f"h{t}"], S5c[key][i][f"fa{t}"], S5c[key][i][f"ms{t}"]] for i in di])
            rows.append((nm, c))
        for nm, c in rows:
            csi = lambda h, fa, ms: h / max(h + fa + ms, 1)
            pod = lambda h, fa, ms: h / max(h + ms, 1)
            far = lambda h, fa, ms: fa / max(h + fa, 1)
            fb = lambda h, fa, ms: (h + fa) / max(h + ms, 1)
            vals = [boot(f, blk, [c[:, 0], c[:, 1], c[:, 2]]) for f in (csi, pod, far, fb)]
            print(f"  {nm:<12}" + "  ".join(f"{lab} {v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}]" for lab, v in zip(("CSI", "POD", "FAR", "頻度比"), vals)))


if __name__ == "__main__":
    main()
