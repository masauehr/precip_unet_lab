"""Phase 4c: 確率の設定ごとの指標、設定間の差の区間（二段ブートストラップ）、メンバー数の影響、
アンサンブルのばらつきと誤差の相関（降水量を除いたもの）を出す。

使い方: .venv/bin/python src/phase4c_report.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boot import PROB_KEYS, PROB_STATS, diff_ci, prob_stats

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase4c"
CFGS = ["基準", "単一", "ens1", "ens2", "ens4", "ens8"]
THRS = ("1", "5", "10")


def load():
    return [np.load(f) for f in sorted(OUT.glob("run_*.npz"))]


def parts(run, cfg, t):
    return {f: run[f"{cfg}|{t}|{f}"] for f in PROB_KEYS}


def partial_corr(x, y, z):
    """z の影響（線形）を除いた x と y の相関。"""
    res = lambda v: v - np.polyval(np.polyfit(z, v, 1), z)
    return float(np.corrcoef(res(x), res(y))[0, 1])


def main():
    runs = load()
    S = len(runs)
    print(f"ラン {S}個（各テスト200件）\n")
    print("[1] 設定ごとの指標（ラン間の 平均 ± 標準偏差）  ens K＝ブートストラップ K メンバーの平均、ens1＝メンバー1個")
    for t in THRS:
        print(f"\n  ■ {t} mm/h 以上")
        print(f"  {'設定':<8}{'BSS':>16}{'CSI':>16}{'頻度バイアス':>16}")
        for c in CFGS:
            v = np.array([[prob_stats(parts(r, c, t))[k] for k in PROB_STATS] for r in runs])
            print(f"  {c:<8}" + "".join(f"{m:+10.3f}±{sd:.3f}" for m, sd in zip(v.mean(0), v.std(0))))

    print("\n[2] 設定間の差の 95%信頼区間（事例のみ ／ 二段＝学習ランのぶれを含む）。* ＝0を含まない")
    pairs = [("単一", "基準"), ("ens8", "単一"), ("ens8", "基準"), ("ens4", "ens2"), ("ens8", "ens4"), ("ens1", "単一")]
    for t in THRS:
        print(f"\n  ■ {t} mm/h 以上")
        print(f"  {'比較':<14}{'指標':<8}{'点推定':>9}{'事例のみ':>22}{'二段':>22}  {'符号一致':>7}")
        for a, b in pairs:
            A, B = [parts(r, a, t) for r in runs], [parts(r, b, t) for r in runs]
            kw = dict(keys=PROB_KEYS, stats=PROB_STATS, stat_fn=prob_stats, n_boot=2000)
            c1, c2 = diff_ci(A, B, "cases", **kw), diff_ci(A, B, "two_level", **kw)
            for k in PROB_STATS[:2]:
                sg = [np.sign(prob_stats(x)[k] - prob_stats(y)[k]) for x, y in zip(A, B)]
                same = max(sum(v > 0 for v in sg), sum(v < 0 for v in sg))
                f = lambda q: f"[{q[1]:+.3f},{q[2]:+.3f}]" + ("*" if (q[1] > 0 or q[2] < 0) else " ")
                print(f"  {a+'−'+b:<12}{k:<8}{c2[k][0]:+9.3f}{f(c1[k]):>22}{f(c2[k]):>22}  {same}/{S}")

    print("\n[3] メンバー数 K の影響（BSS・CSI のラン平均）")
    print(f"  {'しきい値':<8}" + "".join(f"{'K='+k:>16}" for k in ("1", "2", "4", "8")))
    for t in THRS:
        row = f"  {t+'mm/h':<8}"
        for k in ("1", "2", "4", "8"):
            v = np.array([[prob_stats(parts(r, f'ens{k}', t))[x] for x in ("bss", "csi")] for r in runs]).mean(0)
            row += f"{v[0]:+9.3f}/{v[1]:.3f} "
        print(row)

    print("\n[4] アンサンブル(K=8)のばらつきと誤差の相関（事例ごと、ラン平均）")
    rows = {"線形ばらつき×誤差": [], "対数ばらつき×誤差": [], "線形ばらつき×誤差（降水量除去）": [],
            "対数ばらつき×誤差（降水量除去）": [], "降水量×誤差": []}
    for r in runs:
        z = np.log(r["rain"] + 0.05)
        rows["線形ばらつき×誤差"].append(np.corrcoef(r["sp_lin"], r["err"])[0, 1])
        rows["対数ばらつき×誤差"].append(np.corrcoef(r["sp_log"], r["err"])[0, 1])
        rows["線形ばらつき×誤差（降水量除去）"].append(partial_corr(r["sp_lin"], r["err"], z))
        rows["対数ばらつき×誤差（降水量除去）"].append(partial_corr(r["sp_log"], r["err"], z))
        rows["降水量×誤差"].append(np.corrcoef(z, r["err"])[0, 1])
    for k, v in rows.items():
        print(f"  {k:<26}{np.mean(v):+.2f} ± {np.std(v):.2f}")


if __name__ == "__main__":
    main()
