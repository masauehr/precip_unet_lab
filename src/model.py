"""降水量補正用の U-Net。

構成: 3x3畳み込み → GroupNorm → SiLU を2回重ねたブロック、平均プーリングで縮小、
双線形拡大＋スキップ結合で復元。出力は「対数空間の残差 r」で、
予測 = ベース場 × exp(r)（乗法的補正）。最終層をゼロ初期化するので、学習開始時は
ベース場そのもの（r=0）から始まる。ベース場が 0 の所は 0 のまま（無降水を壊さない）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.1  # 対数変換の下駄 [mm/h]。小さすぎると弱雨・ゼロ付近で不安定になる


def _groups(ch: int, max_groups: int = 8) -> int:
    """ch を割り切る最大のグループ数（GroupNorm の制約）。"""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """(3x3畳み込み → GroupNorm → SiLU) × 2。"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(_groups(c_out), c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(_groups(c_out), c_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """入力 (B, c_in, H, W)、H・W は 2**depth で割り切れること。出力 (B, 1, H, W)。"""

    def __init__(self, c_in: int = 1, base: int = 16, depth: int = 3, r_max: float = 3.0):
        super().__init__()
        self.depth = depth
        self.r_max = r_max
        chs = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = c_in
        for ch in chs:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.dec.append(ConvBlock(chs[i] + chs[i - 1], chs[i - 1]))
        self.head = nn.Conv2d(chs[0], 1, 1)
        nn.init.zeros_(self.head.weight)  # 学習開始時は r=0（ベース場のまま）
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < self.depth:
                skips.append(x)
                x = F.avg_pool2d(x, 2)  # 平均プーリング
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skips.pop()], dim=1))  # スキップ結合
        r = self.head(x)
        return self.r_max * torch.tanh(r / self.r_max)  # 補正倍率を exp(±r_max) に制限


class MultiplicativeCorrector(nn.Module):
    """ベース場 x_base (B,1,H,W) [mm/h] → 補正後の降水 (B,1,H,W)。"""

    def __init__(self, base: int = 16, depth: int = 3):
        super().__init__()
        self.unet = UNet(c_in=1, base=base, depth=depth)

    def forward(self, x_base):
        r = self.unet(torch.log(x_base + EPS))  # 対数空間で残差を推定
        return x_base * torch.exp(r)  # 乗法的補正（0 は 0 のまま）


def log_loss(pred, target):
    """対数空間の MSE。強雨が損失を支配するのを防ぐ。"""
    return F.mse_loss(torch.log(pred + EPS), torch.log(target + EPS))


def conservation_loss(pred, coarse, factor: int):
    """粗い格子ごとの平均が、入力の粗い予報と一致するように促す（整数倍の格子）。"""
    return F.mse_loss(F.avg_pool2d(pred, factor), coarse)


def renormalize(pred, coarse, factor: int):
    """粗い格子ごとの平均が coarse に厳密に一致するよう再スケールする（保存性を厳密にする場合）。"""
    cur = F.avg_pool2d(pred, factor)
    ratio = torch.where(cur > 1e-8, coarse / cur.clamp_min(1e-8), torch.zeros_like(cur))
    return pred * F.interpolate(ratio, scale_factor=factor, mode="nearest")
