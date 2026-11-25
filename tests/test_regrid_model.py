"""再格子化とモデルの基本性質のテスト。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regrid import bilinear_matrix, conservative_matrix, regrid_bilinear, regrid_conservative


def test_保存的再格子化は総量を保つ():
    rng = np.random.default_rng(0)
    a = rng.random((30, 45))
    for shape in [(10, 15), (64, 64), (7, 11)]:  # 割り切れない比も含む
        b = regrid_conservative(a, shape)
        assert b.mean() == pytest.approx(a.mean())


def test_保存的再格子化の行は和が1():
    w = conservative_matrix(7, 11)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_整数倍の縮小は面積平均と一致():
    a = np.arange(64, dtype=float).reshape(8, 8)
    b = regrid_conservative(a, (4, 4))
    assert b[0, 0] == pytest.approx(a[:2, :2].mean())


def test_双線形は定数場を保つが総量は保存しない例がある():
    a = np.ones((5, 5))
    assert np.allclose(regrid_bilinear(a, (13, 13)), 1.0)
    assert np.allclose(bilinear_matrix(5, 13).sum(axis=1), 1.0)
    # 粗くする向きでは、尖った場の総量が変わる（降水に双線形を使わない理由）
    p = np.zeros((6, 6))
    p[2, 2] = 10.0
    assert regrid_bilinear(p, (4, 4)).mean() != pytest.approx(p.mean())
    assert regrid_conservative(p, (4, 4)).mean() == pytest.approx(p.mean())


torch = pytest.importorskip("torch")
from model import MultiplicativeCorrector, UNet, renormalize


def test_初期状態はベース場そのもの():
    m = MultiplicativeCorrector(base=8, depth=3)
    x = torch.rand(2, 1, 32, 32) * 5
    x[:, :, :8] = 0.0
    assert torch.allclose(m(x), x)


def test_ゼロは補正後もゼロ():
    m = MultiplicativeCorrector(base=8, depth=3)
    for p in m.parameters():  # 重みを乱して補正を効かせる
        torch.nn.init.normal_(p, std=0.1)
    x = torch.zeros(1, 1, 32, 32)
    x[:, :, 10:20, 10:20] = 3.0
    y = m(x)
    assert (y[x == 0] == 0).all() and (y >= 0).all()


def test_補正倍率は上限内():
    u = UNet(base=8, depth=2, r_max=3.0)
    for p in u.parameters():
        torch.nn.init.normal_(p, std=1.0)
    r = u(torch.rand(1, 1, 16, 16))
    assert r.abs().max() <= 3.0 + 1e-6


def test_再スケールで粗い格子平均が一致():
    pred = torch.rand(2, 1, 32, 32)
    coarse = torch.rand(2, 1, 4, 4)
    out = renormalize(pred, coarse, 8)
    assert torch.allclose(torch.nn.functional.avg_pool2d(out, 8), coarse, atol=1e-5)
