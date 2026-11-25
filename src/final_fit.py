"""最終評価の準備: 最終モデルの学習と、後処理（QM・τ・階級別基準）の固定。**テスト集合は読まない**。

- 学習: 学習＋検証の297日すべてで、補正モデル（Phase 5b と同じ設定）と確率モデル（Phase 5d と同じ設定）を各3シード、10エポック。
  重みを outputs/final/ に保存する（学習時のログは検証損失を出さない・設定は交差検証で固定済みのものから変えない）。
- 較正: 交差検証の予測（OOF）で、QM の写像（基準・補正モデル各シード）、確率の二値化 τ（アンサンブル・階級別基準）、
  気候値（発生率）、階級別基準の当てはめ用標本を作って outputs/final/calib.pkl に保存する。
使い方（学習は種類×シードごとに別プロセス）:
  .venv/bin/python src/final_fit.py --train hybrid --seed 0   （hybrid / prob）
  .venv/bin/python src/final_fit.py --calib
最終評価は src/final_eval.py（1回だけ）。方針は FINAL_PROTOCOL.md。
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5d
from model import PROB_THRS
from phase5b import OUT as OUT_B, build
from phase5c import apply_qm, fit_qm
from phase5d import OUT as OUT_D
from prob import best_tau
from real_model import load_set, train as train_hybrid
from split import make_split

OUT = Path(__file__).resolve().parents[1] / "outputs" / "final"
GRID = np.linspace(0.05, 0.95, 19)
SUB = 4_000_000
SEEDS = (0, 1, 2)
EPOCHS = 10


def load_all():
    sets = make_split()[0]
    x1, y1, b1 = load_set(sets["train"], "cache_train.npz")
    x2, y2, b2 = load_set(sets["val"], "cache_val.npz")
    return tuple(torch.cat(p) for p in ((x1, x2), (y1, y2), (b1, b2)))


def fit_models(kind, seed, threads):
    torch.set_num_threads(threads)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{kind}_seed{seed}.pt"
    if f.exists():
        print("取得済み", f.name)
        return
    x, y, bad = load_all()
    t0 = time.time()
    log = lambda m: print(f"  {m}  経過 {time.time() - t0:.0f}s", flush=True)
    if kind == "hybrid":
        model = train_hybrid(x, y, bad, seed=seed, epochs=EPOCHS, log=log)
    else:
        model = phase5d.train(x, y, bad, seed, EPOCHS, log=log)
    torch.save(model.state_dict(), f)
    print("保存", f.name, flush=True)


def calib():
    """OOF から後処理を固定する（テストは使わない）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    days, blocks, fold = build(6)
    fold_arr = np.repeat(np.array(fold), 24)
    x, y, bad = load_all()
    X, Y = x.numpy()[:, 0], y.numpy()[:, 0]
    OK = ~bad.numpy()[:, 0]
    rng = np.random.default_rng(0)
    C = {"qm_base": fit_qm(X, Y, OK), "qm_model": {}, "prob": {}}
    for s in SEEDS:
        Pm = np.zeros_like(X)
        for k in range(6):
            Pm[fold_arr == k] = np.load(OUT_B / f"pred_seed{s}_fold{k}.npz")["pred"][:, 0].astype(np.float32)
        C["qm_model"][s] = fit_qm(Pm, Y, OK)
    P = {s: np.zeros((len(X), 3, 80, 100), np.float16) for s in SEEDS}
    for s in SEEDS:
        for k in range(6):
            P[s][fold_arr == k] = np.load(OUT_D / f"seed{s}_fold{k}.npz")["p"]
    fx = X[OK]
    sub = rng.choice(len(fx), size=min(SUB, len(fx)), replace=False)
    for j, t in enumerate(PROB_THRS):
        fo = (Y >= t)[OK].astype(np.float32)
        ens = np.mean([P[s][:, j].astype(np.float32) for s in SEEDS], axis=0)[OK]
        C["prob"][t] = {"clim": float(fo.mean()), "tau_ens": best_tau(ens[sub], fo[sub], GRID),
                        "base_x": fx[sub], "base_o": fo[sub]}
        print(f"≥{t:g}: 発生率 {fo.mean():.4f}  τ(アンサンブル) {C['prob'][t]['tau_ens']:.2f}", flush=True)
    # 階級別基準の τ（基準自身の当てはめ上で CSI 最大）
    from prob import binned_baseline
    for t in PROB_THRS:
        c = C["prob"][t]
        c["tau_base"] = best_tau(binned_baseline(c["base_x"], c["base_o"], c["base_x"]), c["base_o"], GRID)
    with open(OUT / "calib.pkl", "wb") as f:
        pickle.dump(C, f)
    print("保存 calib.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["hybrid", "prob"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--calib", action="store_true")
    a = ap.parse_args()
    if a.calib:
        calib()
    elif a.train:
        fit_models(a.train, a.seed, a.threads)
