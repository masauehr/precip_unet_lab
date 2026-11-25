"""Phase 4b: 超過確率とブートストラップ・アンサンブル。

比べる:
  基準の確率   : ベース場を階級に分け、学習データの階級別の発生率を確率とする（素の予報を素直に確率へ）
  単一ネット   : 量（Tweedie）＋しきい値 1・5・10 mm/h の超過確率（BCE）。全学習データで1回学習
  アンサンブル : 学習データを事例単位で復元抽出（ブートストラップ）した K 個のネットの平均
指標: ブライアススキル（気候値に対して）・信頼性の乱れ ECE・信頼性図、学習データで選んだ確率しきい値での CSI と頻度バイアス、
      量の指標（対数RMSE・総量比・雨画素RMSE）、アンサンブルのばらつきと誤差の相関。
学習シード 100〜（Phase 3・4 と同じ新シード）。各テスト200件。

使い方: .venv/bin/python src/phase4b.py [--runs 3] [--members 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prob
from model import PROB_THRS, ProbCorrector, prob_loss
from phase1b import metrics
from regrid import regrid_conservative
from synth import make_dataset
from train import FACTOR, N


def train_prob(x, y, seed, epochs=15, boot=False):
    """x, y: (n,1,H,W) テンソル。boot=True なら事例を復元抽出して学習する。"""
    torch.manual_seed(seed)
    n = len(x)
    if boot:
        idx = torch.tensor(np.random.default_rng(seed).integers(0, n, size=n))
        x, y = x[idx], y[idx]
    m = ProbCorrector(win=2 * FACTOR + 1)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 16):
            b = perm[i : i + 16]
            mu, z = m.heads(x[b])
            loss = prob_loss(mu, z, y[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m.eval()


def infer(m, x):
    with torch.no_grad():
        mu, p = m(x)
    return mu.numpy()[:, 0], p.numpy()  # (n,H,W), (n,K,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--members", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(4)
    T = lambda z: torch.tensor(z[:, None], dtype=torch.float32)

    keys = ["基準の確率", "単一ネット", "アンサンブル"]
    out = {k: {t: [] for t in PROB_THRS} for k in keys}   # (BSS, ECE, CSI, fbias)
    amount = {"単一ネット": [], "アンサンブル": []}
    rel = {t: {k: [] for k in keys} for t in PROB_THRS}   # 信頼性図用の (p, o)
    spread_err = []

    for r in range(a.runs):
        s = 100 + r
        c_tr, f_tr = make_dataset(400, N, FACTOR, seed=s)
        c_te, f_te = make_dataset(200, N, FACTOR, seed=20_000 + s)
        b_tr, b_te = regrid_conservative(c_tr, (N, N)), regrid_conservative(c_te, (N, N))
        x_tr, y_tr, x_te = T(b_tr), T(f_tr), T(b_te)

        single = train_prob(x_tr, y_tr, s)
        mu1, p1 = infer(single, x_te)
        _, p1_tr = infer(single, x_tr)
        members = [train_prob(x_tr, y_tr, 1000 * (r + 1) + j, boot=True) for j in range(a.members)]
        outs = [infer(m, x_te) for m in members]
        outs_tr = [infer(m, x_tr)[1] for m in members]
        muE = np.mean([o[0] for o in outs], axis=0)
        pE, pE_tr = np.mean([o[1] for o in outs], axis=0), np.mean(outs_tr, axis=0)

        preds = {
            "単一ネット": (prob.monotone([p1[:, k] for k in range(len(PROB_THRS))]),
                       prob.monotone([p1_tr[:, k] for k in range(len(PROB_THRS))])),
            "アンサンブル": (prob.monotone([pE[:, k] for k in range(len(PROB_THRS))]),
                         prob.monotone([pE_tr[:, k] for k in range(len(PROB_THRS))])),
        }
        for k, t in enumerate(PROB_THRS):
            o_te, o_tr = (f_te >= t).astype(float), (f_tr >= t).astype(float)
            clim = o_tr.mean()
            cand = {"基準の確率": (prob.binned_baseline(b_tr.ravel(), o_tr.ravel(), b_te.ravel()).reshape(o_te.shape),
                               prob.binned_baseline(b_tr.ravel(), o_tr.ravel(), b_tr.ravel()).reshape(o_tr.shape))}
            for name in ("単一ネット", "アンサンブル"):
                cand[name] = (preds[name][0][k], preds[name][1][k])
            for name, (pt, ptr) in cand.items():
                tau = prob.best_tau(ptr.ravel(), o_tr.ravel())
                pr = pt >= tau
                h, fa, ms = (pr & (o_te > 0)).sum(), (pr & (o_te == 0)).sum(), (~pr & (o_te > 0)).sum()
                out[name][t].append((prob.bss(pt.ravel(), o_te.ravel(), clim), prob.ece(pt.ravel(), o_te.ravel()),
                                     h / max(h + fa + ms, 1), (h + fa) / max(h + ms, 1)))
                rel[t][name].append((pt.ravel(), o_te.ravel()))
        for name, mu in (("単一ネット", mu1), ("アンサンブル", muE)):
            m_ = metrics(mu, f_te)
            amount[name].append((m_["rmse"].mean(), m_["total"], m_["wet_rmse"], m_["strong"]))
        # アンサンブルのばらつき（量の標準偏差）と、アンサンブル平均の誤差の相関（事例ごと）
        sp = np.std([o[0] for o in outs], axis=0).mean(axis=(1, 2))
        er = np.sqrt(((np.log(muE + 0.1) - np.log(f_te + 0.1)) ** 2).mean(axis=(1, 2)))
        spread_err.append(np.corrcoef(sp, er)[0, 1])
        print(f"ラン {s} 完了", flush=True)

    print(f"\n学習ラン{a.runs}回（アンサンブルは各 {a.members} メンバー）、テスト各200件の平均")
    print("確率は、しきい値以上になる確率。BSS＝気候値に対するブライアススキル（大きいほど良い）、")
    print("ECE＝信頼性の乱れ（小さいほど良い）、CSI/頻度バイアスは、学習データで CSI 最大の確率しきい値で二値化")
    for t in PROB_THRS:
        print(f"\n[{t:g} mm/h 以上]  発生率(学習)≈{np.mean([(f_tr >= t).mean()]):.3f}")
        print(f"{'設定':<12}{'BSS':>8}{'ECE':>8}{'CSI':>8}{'頻度バイアス':>12}")
        for n in keys:
            v = np.array(out[n][t]).mean(axis=0)
            print(f"{n:<10}{v[0]:+8.3f}{v[1]:8.4f}{v[2]:8.3f}{v[3]:12.2f}")
    print("\n[量の平均予測（Tweedie）]")
    print(f"{'設定':<12}{'対数RMSE':>9}{'総量比':>8}{'雨画素RMSE':>11}{'強雨比A':>8}")
    for n, v in amount.items():
        m_ = np.mean(v, axis=0)
        print(f"{n:<10}{m_[0]:9.3f}{m_[1]:+8.2f}{m_[2]:11.3f}{m_[3]:+8.2f}")
    print(f"\nアンサンブルのばらつき（量の標準偏差の事例平均）と誤差の相関: {np.mean(spread_err):+.2f}（ラン平均）")
    t = 5.0
    print(f"\n[信頼性図 5 mm/h 以上。予測確率の帯 → 観測頻度（件数）。対角線＝理想]")
    print(f"{'帯':<10}" + "".join(f"{n:>26}" for n in keys))
    tab = {n: prob.reliability(np.concatenate([p for p, _ in rel[t][n]]), np.concatenate([o for _, o in rel[t][n]]), 10) for n in keys}
    for b in range(10):
        row = f"{b/10:.1f}-{(b+1)/10:.1f}  "
        for n in keys:
            pm, om, cnt = tab[n][b]
            row += f"{('p=%.2f obs=%.2f n=%d' % (pm, om, cnt)) if cnt else '-':>26}"
        print(row)


if __name__ == "__main__":
    main()
