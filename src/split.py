"""実データ（GSM×GSMaP）の学習・検証・テストの分割（ブロック巡回）。

決め方（結果を見る前に固定する。見てから変えない）:
- 2025-04-01 を 0 日目とし、11日を1周期＝「10日のブロック＋空け1日」とする。空けの日はどこにも入れない（隣のブロックとの相関の漏れを避ける）。
- ブロックを 0,1,2,… と番号づけ、10ブロックの繰り返しで役割を割り当てる（学習7・検証1・テスト2）:
    ブロック番号 mod 10 = 5 → 検証、2 または 7 → テスト、それ以外 → 学習
  テストは5ブロックおきに散らばり、季節の偏りが出にくい。
- 事例の単位は日（24時間分）。ブートストラップはブロック単位（block_id）で行う。
使い方: .venv/bin/python src/split.py
"""
from datetime import date
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "data"
ORIGIN = date(2025, 4, 1)
BLOCK_DAYS = 10
PERIOD = BLOCK_DAYS + 1
VAL_MOD = (5,)
TEST_MOD = (2, 7)


def locate(d: date):
    """日付 → (ブロック番号, ブロック内の位置 0..10)。位置10が空けの日。"""
    n = (d - ORIGIN).days
    return n // PERIOD, n % PERIOD


def role(d: date) -> str:
    """日付の役割: 'train' / 'val' / 'test' / 'gap'。"""
    b, pos = locate(d)
    if pos == BLOCK_DAYS:
        return "gap"
    m = b % 10
    return "val" if m in VAL_MOD else "test" if m in TEST_MOD else "train"


def valid_days():
    """GSM・GSMaP がそろい、GSMaP が全面 -99 でない日（YYYYMMDD の一覧）。"""
    out = []
    for p in sorted((D / "gsmap").glob("2*.npy")):
        if (D / "gsm" / p.name).exists() and not (np.load(p) < -90).all():
            out.append(p.stem)
    return out


def make_split(days=None):
    """{'train': [...], 'val': [...], 'test': [...]}（YYYYMMDD）と、日→ブロック番号の辞書を返す。"""
    days = valid_days() if days is None else days
    sets = {"train": [], "val": [], "test": []}
    block = {}
    for s in days:
        d = date(int(s[:4]), int(s[4:6]), int(s[6:]))
        r = role(d)
        if r != "gap":
            sets[r].append(s)
            block[s] = locate(d)[0]
    return sets, block


if __name__ == "__main__":
    sets, block = make_split()
    print("集合  日数 ブロック数  月ごとの日数")
    for k, v in sets.items():
        mon = {}
        for s in v:
            mon[s[:6]] = mon.get(s[:6], 0) + 1
        print(f"{k:<6}{len(v):>4}{len({block[s] for s in v}):>8}  " + " ".join(f"{m[2:]}:{n}" for m, n in sorted(mon.items())))
    print("\n集合ごとの強雨（GSMaP の ≥10 mm/h 画素数・≥5 mm/h 画素数・独立エピソードの目安）")
    for k, v in sets.items():
        a10 = a5 = 0
        big = []
        for s in v:
            a = np.load(D / "gsmap" / f"{s}.npy")
            a = np.where(a < 0, np.nan, a)
            c10 = int(np.nansum(a >= 10))
            a10 += c10
            a5 += int(np.nansum(a >= 5))
            big.append(c10 >= 100)
        print(f"{k:<6} ≥10: {a10:>8,}  ≥5: {a5:>9,}  強雨日(≥10mm/hが100画素以上): {sum(big)}日")
