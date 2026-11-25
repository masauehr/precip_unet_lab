"""二段ブートストラップのテスト。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boot import KEYS, case_index, diff_ci, stats_from_sums


def make_parts(rng, n=100, shift=0.0, seed_effect=0.0):
    """合成の部分和。shift は rmse の底上げ、seed_effect はシード全体に共通のずれ。"""
    return dict(
        rmse=rng.normal(0.9, 0.2, n) + shift + seed_effect,
        sp=rng.uniform(1, 2, n), st=rng.uniform(1, 2, n),
        sps=rng.uniform(1, 2, n), sts=rng.uniform(1, 2, n),
        wsq=rng.uniform(1, 2, n), wn=np.full(n, 1.0),
        fnum=rng.uniform(1, 2, n), fden=rng.uniform(3, 4, n),
    )


def test_ブロック再標本化は連続した事例を含む():
    idx = case_index(np.random.default_rng(0), 100, block=5)
    assert len(idx) == 100
    assert (np.diff(idx.reshape(-1, 5), axis=1) == 1).all()


def test_部分和から指標が出る():
    p = {k: np.ones(10) for k in KEYS}
    st = stats_from_sums(p)
    assert st["total"] == pytest.approx(0.0) and st["wet_rmse"] == pytest.approx(1.0)
    assert st["fss5"] == pytest.approx(0.0)


def test_差がなければ区間は0を含み_差があれば含まない():
    rng = np.random.default_rng(1)
    A = [make_parts(rng) for _ in range(6)]
    B = [{k: v.copy() for k, v in p.items()} for p in A]
    ci = diff_ci(A, B, "two_level", n_boot=300)
    assert ci["rmse"][1] <= 0 <= ci["rmse"][2]
    C = [{**p, "rmse": p["rmse"] + 0.3} for p in A]
    ci = diff_ci(C, B, "two_level", n_boot=300)
    assert ci["rmse"][1] > 0


def test_シード間のぶれがあると二段の区間は事例のみより広い():
    rng = np.random.default_rng(2)
    # 各シードの A−B の差が、シードごとに大きくぶれる（学習のぶれ）
    A, B = [], []
    for _ in range(8):
        eff = rng.normal(0, 0.1)
        A.append(make_parts(rng, shift=eff))
        B.append(make_parts(rng))
    w = lambda ci: ci["rmse"][2] - ci["rmse"][1]
    assert w(diff_ci(A, B, "two_level", n_boot=400)) > 2 * w(diff_ci(A, B, "cases", n_boot=400))
