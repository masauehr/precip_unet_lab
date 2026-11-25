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


from model import TwoStageCorrector, two_stage_loss


def test_2段モデルの形と確率():
    m = TwoStageCorrector(base=8, depth=3)
    x = torch.rand(2, 1, 32, 32) * 5
    amount, p = m(x)
    assert amount.shape == p.shape == (2, 1, 32, 32)
    assert ((p > 0) & (p < 1)).all()
    assert torch.allclose(amount, x)  # 量は初期状態でベース場そのもの


def test_2段損失は逆伝播できる():
    m = TwoStageCorrector(base=8, depth=2)
    x = torch.rand(2, 1, 16, 16) * 5
    t = (torch.rand(2, 1, 16, 16) * 3) * (torch.rand(2, 1, 16, 16) > 0.5)
    amount, logit = m.logits(x)
    loss = two_stage_loss(amount, logit, t)
    loss.backward()
    assert torch.isfinite(loss)


from model import HybridCorrector


def test_ハイブリッドは初期状態でほぼベース場():
    m = HybridCorrector(base=8, depth=3)
    x = torch.rand(2, 1, 32, 32) * 5
    x[:, :, :8] = 0.0
    amount, p = m(x)
    assert amount.shape == p.shape == (2, 1, 32, 32)
    assert torch.allclose(amount, x, atol=0.01)  # 加法の項は初期状態でほぼ0
    assert (amount[:, :, :8] > 0).all()  # ベース場0でも、加法の項で正の値を出せる


def test_ハイブリッドの加法なしは乗法のみ():
    m = HybridCorrector(base=8, depth=2, additive=False, neighbor=False)
    x = torch.rand(1, 1, 16, 16) * 5
    x[:, :, :4] = 0.0
    amount, _ = m(x)
    assert (amount[:, :, :4] == 0).all()


from model import gamma_deviance_loss, tweedie_loss


def test_偏差損失は平均で最小になる():
    # 同じ予測 μ を、分布の揃った真値の標本に当てたとき、損失の最小は標本平均付近（対数MSEは中央値寄り）
    y = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 20.0]).reshape(1, 1, 2, 3)
    grid = torch.linspace(0.3, 12.0, 400)
    loss = torch.stack([tweedie_loss(torch.full_like(y, float(m)), y) for m in grid])
    best = float(grid[loss.argmin()])
    assert abs(best - float(y.mean())) < 0.2


def test_ガンマ偏差は一致で0():
    y = torch.tensor([[[[2.0, 5.0]]]])
    logit = torch.full_like(y, 20.0)  # 雨ありを確信
    assert float(gamma_deviance_loss(y.clone(), logit, y)) < 1e-6


from train import fss


def test_UNetのオプションが動く():
    x = torch.rand(1, 3, 32, 32)
    for kw in [dict(norm="none"), dict(pool="max"), dict(skip=False), dict(up="nearest"), dict(depth=2), dict(depth=4)]:
        u = UNet(c_in=3, base=8, **{"depth": 3, **kw}, c_out=3)
        assert u(x).shape == (1, 3, 32, 32)
    m = HybridCorrector(base=8, depth=3, skip=False, norm="none")
    a, p = m(torch.rand(1, 1, 32, 32))
    assert a.shape == (1, 1, 32, 32)


def test_FSSは一致で1_ずれても近ければ高い():
    t = np.zeros((1, 32, 32)); t[0, 10:14, 10:14] = 5.0
    assert fss(t, t, 1.0) == pytest.approx(1.0)
    shifted = np.roll(t, 3, axis=2)
    far = np.roll(t, 15, axis=2)
    assert fss(shifted, t, 1.0) > fss(far, t, 1.0)


from metrics_ext import categorical, cond_ratios, fss_window, spectrum_ratio


def test_拡張指標の基本性質():
    t = np.zeros((2, 32, 32)); t[:, 10:14, 10:14] = 6.0
    c = categorical(t, t, 5.0)
    assert c["csi"] == 1.0 and c["fbias"] == 1.0
    assert fss_window(t, t, 5.0, 1) == pytest.approx(1.0)
    # 位置がずれると、窓が大きいほど FSS は高い
    s = np.roll(t, 6, axis=2)
    assert fss_window(s, t, 5.0, 17) > fss_window(s, t, 5.0, 1)
    # 完全な予測は、どちらの条件付けでも比は0（log 1）
    a, b = cond_ratios(t, t, 5.0)
    assert a == pytest.approx(0.0) and b == pytest.approx(0.0)


def test_平滑化すると細かい波数のパワーが減る():
    rng = np.random.default_rng(0)
    t = rng.random((4, 32, 32))
    sm = (t + np.roll(t, 1, 1) + np.roll(t, 1, 2) + np.roll(np.roll(t, 1, 1), 1, 2)) / 4
    r = spectrum_ratio(sm, t, bands=((1, 4), (9, 16)))
    assert r[1] < r[0] < 1.01
