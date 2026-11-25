"""Phase 3: 収集した部分和から、設定ごとの指標とシード間のぶれ、既定との差の信頼区間（3方式）を出す。

使い方: .venv/bin/python src/phase3_report.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boot import KEYS, STATS, diff_ci, stats_from_sums

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase3"
NAMES = ["既定", "power1.2", "最大プーリング", "加法なし(乗法のみ)"]


def load(name):
    files = sorted(OUT.glob(f"{name}_s*.npz"), key=lambda p: int(p.stem.split("_s")[-1]))
    return {int(f.stem.split("_s")[-1]): {k: np.load(f)[k] for k in KEYS} for f in files}


def main():
    data = {n: load(n) for n in NAMES}
    seeds = sorted(set.intersection(*[set(d) for d in data.values()]))
    print(f"対象シード {len(seeds)}個: {seeds}（各テスト200件）\n")
    A = {n: [data[n][s] for s in seeds] for n in NAMES}

    print("[1] 設定ごとの指標（シード間の 平均 ± 標準偏差）")
    print(f"{'設定':<16}" + "".join(f"{k:>16}" for k in STATS))
    for n in NAMES:
        per = np.array([[stats_from_sums(p)[k] for k in STATS] for p in A[n]])
        print(f"{n:<14}" + "".join(f"{m:+10.3f}±{sd:.3f}" for m, sd in zip(per.mean(0), per.std(0))))

    print("\n[2] 既定との差（設定 − 既定）の 95%信頼区間。3方式で幅を比べる（0を含まなければ差あり）")
    print("    事例のみ＝これまでの表の区間に相当、二段＝学習のぶれを含む")
    for n in NAMES[1:]:
        print(f"\n  ■ {n} − 既定")
        print(f"  {'指標':<10}{'点推定':>9}{'事例のみ':>22}{'シードのみ':>22}{'二段':>22}  {'符号一致(シード)':>14}")
        res = {m: diff_ci(A[n], A["既定"], m, n_boot=2000) for m in ["cases", "seeds", "two_level"]}
        for k in STATS:
            sg = [np.sign(stats_from_sums(a)[k] - stats_from_sums(b)[k]) for a, b in zip(A[n], A["既定"])]
            same = max(sum(x > 0 for x in sg), sum(x < 0 for x in sg))
            fmt = lambda t: f"[{t[1]:+.3f},{t[2]:+.3f}]"
            sig = lambda t: "*" if (t[1] > 0 or t[2] < 0) else " "
            r = res
            print(f"  {k:<10}{r['two_level'][k][0]:+9.3f}{fmt(r['cases'][k]):>21}{sig(r['cases'][k])}"
                  f"{fmt(r['seeds'][k]):>21}{sig(r['seeds'][k])}{fmt(r['two_level'][k]):>21}{sig(r['two_level'][k])}"
                  f"  {same}/{len(sg)}")
    print("\n  * ＝ 区間が0を含まない")


if __name__ == "__main__":
    main()
