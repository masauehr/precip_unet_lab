"""確率評価のテスト。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prob import best_tau, binned_baseline, brier, bss, ece, monotone, reliability


def test_完全な確率はブライアス0_気候値はスキル0():
    o = np.array([0, 1, 0, 1.0])
    assert brier(o, o) == 0.0
    assert bss(np.full(4, 0.5), o, 0.5) == pytest.approx(0.0)


def test_信頼性が良ければECEは小さい():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    good = (rng.random(200_000) < p).astype(float)  # 予測確率どおりに起きる
    assert ece(p, good) < 0.01
    assert ece(np.clip(p * 0.5, 0, 1), good) > 0.1  # 確率が低すぎる
    assert sum(r[2] for r in reliability(p, good)) == 200_000


def test_単調化():
    a, b = monotone([np.array([0.2, 0.5]), np.array([0.4, 0.1])])
    assert (b <= a).all() and b[0] == 0.2


def test_階級別の基準は階級内の発生率():
    rng = np.random.default_rng(1)
    base = rng.random(20_000) * 10
    o = (rng.random(20_000) < base / 10).astype(float)
    p = binned_baseline(base, o, np.array([0.5, 9.5]))
    assert p[0] < 0.2 and p[1] > 0.8


def test_最良しきい値は範囲内():
    rng = np.random.default_rng(2)
    o = (rng.random(5000) < 0.2).astype(float)
    p = np.clip(o * 0.6 + rng.random(5000) * 0.3, 0, 1)
    assert 0.05 <= best_tau(p, o) <= 0.95
