"""分割（ブロック巡回）の性質のテスト。データ不要。"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from split import ORIGIN, locate, make_split, role


def _all_days(n=400):
    return [f"{ORIGIN + timedelta(days=i):%Y%m%d}" for i in range(n)]


def test_空けの日はどこにも入らない():
    sets, _ = make_split(_all_days())
    used = {s for v in sets.values() for s in v}
    assert len(used) == sum(len(v) for v in sets.values())  # 重複なし
    for i in range(400):
        d = ORIGIN + timedelta(days=i)
        if locate(d)[1] == 10:
            assert f"{d:%Y%m%d}" not in used


def test_同じブロックは同じ集合():
    sets, block = make_split(_all_days())
    role_of = {s: k for k, v in sets.items() for s in v}
    seen = {}
    for s, b in block.items():
        assert seen.setdefault(b, role_of[s]) == role_of[s]


def test_隣り合う別集合のブロックの間に空けがある():
    # 日付の並びで集合が切り替わる所には、必ず空けの日が1日以上ある
    prev = None
    for i in range(400):
        r = role(ORIGIN + timedelta(days=i))
        if r != "gap":
            if prev not in (None, "gap") and prev != r:
                raise AssertionError("集合が空けなしで切り替わっている")
        prev = r


def test_ブロック数の比は学習7_検証1_テスト2():
    sets, block = make_split(_all_days(110 * 5))
    n = {k: len({block[s] for s in v}) for k, v in sets.items()}
    assert (n["train"], n["val"], n["test"]) == (35, 5, 10)


def test_GSMをGSMaP格子に載せても総量が保たれ定数場は定数のまま():
    import numpy as np
    from real_data import gsm_on_gsmap
    one = gsm_on_gsmap(np.ones((2, 81, 81)))
    assert np.allclose(one, 1.0, atol=1e-5) and one.shape == (2, 80, 100)
    # 点（1画素）だけの降水量の総量: GSM 側の面積重み付き総量と一致
    a = np.zeros((1, 81, 81)); a[0, 40, 40] = 1.0
    out = gsm_on_gsmap(a)
    assert abs(out.sum() * 0.1 * 0.1 - 0.1 * 0.125) < 1e-6
