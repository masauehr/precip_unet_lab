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

    def __init__(self, c_in: int = 1, base: int = 16, depth: int = 3, r_max: float = 3.0, c_out: int = 1):
        super().__init__()
        self.depth = depth
        self.c_out = c_out
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
        self.head = nn.Conv2d(chs[0], c_out, 1)
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
        r0 = self.r_max * torch.tanh(r[:, :1] / self.r_max)  # 補正倍率を exp(±r_max) に制限
        if self.c_out == 1:
            return r0
        return torch.cat([r0, r[:, 1:]], dim=1)  # 2ch目以降は制限しない（発生のロジットなど）


class MultiplicativeCorrector(nn.Module):
    """ベース場 x_base (B,1,H,W) [mm/h] → 補正後の降水 (B,1,H,W)。"""

    def __init__(self, base: int = 16, depth: int = 3):
        super().__init__()
        self.unet = UNet(c_in=1, base=base, depth=depth)

    def forward(self, x_base):
        r = self.unet(torch.log(x_base + EPS))  # 対数空間で残差を推定
        return x_base * torch.exp(r)  # 乗法的補正（0 は 0 のまま）


class TwoStageCorrector(nn.Module):
    """発生の有無と量を分ける2段モデル。

    量   : amount = ベース場 × exp(r)（雨があるときの量。乗法補正）
    発生 : p = sigmoid(z)（その画素に雨がある確率）
    出力 : (amount, p)。期待値の予測は amount × p、地図としての予測は amount × (p>0.5)。
    """

    def __init__(self, base: int = 16, depth: int = 3):
        super().__init__()
        self.unet = UNet(c_in=1, base=base, depth=depth, c_out=2)

    def forward(self, x_base):
        out = self.unet(torch.log(x_base + EPS))
        amount = x_base * torch.exp(out[:, :1])
        return amount, torch.sigmoid(out[:, 1:])

    def logits(self, x_base):
        """BCE 用に、sigmoid 前のロジットを返す。"""
        out = self.unet(torch.log(x_base + EPS))
        return x_base * torch.exp(out[:, :1]), out[:, 1:]


class HybridCorrector(nn.Module):
    """乗法＋加法のハイブリッド（発生と量を分ける2段）。

    量   : amount = ベース場 × exp(r) ＋ softplus(a)   … 加法の項で、ベース場が0の所にも雨を出せる
    発生 : p = sigmoid(z)
    入力 : log(ベース場+ε)。neighbor=True なら、近傍（窓 2*factor+1）の最大と平均の対数も加える
           （粗い予報が隣のセルにある雨＝位置ずれを拾うため）。
    加法の項は、学習開始時はほぼ0（バイアス −6）にして、乗法補正から始める。
    出力チャンネル: [r, z] または [r, z, a]
    """

    def __init__(self, base: int = 16, depth: int = 3, additive: bool = True,
                 neighbor: bool = True, win: int = 17):
        super().__init__()
        self.additive, self.neighbor, self.win = additive, neighbor, win
        c_in = 3 if neighbor else 1
        self.unet = UNet(c_in=c_in, base=base, depth=depth, c_out=3 if additive else 2)
        if additive:
            with torch.no_grad():
                self.unet.head.bias[2] = -6.0  # softplus(-6) ≈ 0.0025 mm/h

    def _features(self, x):
        feats = [torch.log(x + EPS)]
        if self.neighbor:
            pad = self.win // 2
            mx = F.max_pool2d(x, self.win, stride=1, padding=pad)
            av = F.avg_pool2d(x, self.win, stride=1, padding=pad, count_include_pad=False)
            feats += [torch.log(mx + EPS), torch.log(av + EPS)]
        return torch.cat(feats, dim=1)

    def logits(self, x_base):
        out = self.unet(self._features(x_base))
        amount = x_base * torch.exp(out[:, :1])
        if self.additive:
            amount = amount + F.softplus(out[:, 2:3])
        return amount, out[:, 1:2]

    def forward(self, x_base):
        amount, logit = self.logits(x_base)
        return amount, torch.sigmoid(logit)


WET = 0.1  # この値以上を「雨あり」とみなす [mm/h]


def two_stage_loss(amount, logit, target, eps: float = EPS, w_strong: float = 0.0):
    """発生の BCE ＋ 雨のある画素だけの量の対数MSE。

    w_strong>0 なら、真値が 5 mm/h 以上の画素の量の誤差に (1+w_strong) 倍の重みを付ける。
    """
    wet = (target >= WET).float()
    bce = F.binary_cross_entropy_with_logits(logit, wet)
    d = (torch.log(amount + eps) - torch.log(target + eps)) ** 2
    w = wet * (1.0 + w_strong * (target >= 5.0).float())
    amt = (d * w).sum() / w.sum().clamp_min(1.0)
    return bce + amt


MU_MIN = 0.01  # 偏差損失で予測を割る・平方根を取るときの下限 [mm/h]


def gamma_deviance_loss(amount, logit, target):
    """発生の BCE ＋ 雨のある画素だけの量のガンマ偏差。

    ガンマ偏差 d = y/μ − log(y/μ) − 1 は、予測 μ を**条件付きの平均**に導く（対数MSEは中央値寄り）。
    """
    wet = (target >= WET).float()
    bce = F.binary_cross_entropy_with_logits(logit, wet)
    mu = amount.clamp_min(MU_MIN)
    y = target.clamp_min(WET)
    d = y / mu - torch.log(y / mu) - 1.0
    return bce + (d * wet).sum() / wet.sum().clamp_min(1.0)


def tweedie_loss(mu, target, power: float = 1.5):
    """Tweedie 偏差（1<power<2）。0 を含む降水に自然で、予測 μ を条件付きの平均に導く。全画素で使う。"""
    mu = mu.clamp_min(MU_MIN)
    a, b = 2.0 - power, 1.0 - power
    d = 2.0 * (target.pow(a) / (b * a) - target * mu.pow(b) / b + mu.pow(a) / a)
    return d.mean()


def log_loss(pred, target, eps: float = EPS):
    """対数空間の MSE。強雨が損失を支配するのを防ぐ。eps は対数変換の下駄。"""
    return F.mse_loss(torch.log(pred + eps), torch.log(target + eps))


def coarse_total_loss(pred, target, factor: int, eps: float = EPS):
    """粗い格子ごとの平均を、真値の粗い平均に合わせる（対数空間）。対数損失で総量が落ちるのを抑える。"""
    return F.mse_loss(
        torch.log(F.avg_pool2d(pred, factor) + eps),
        torch.log(F.avg_pool2d(target, factor) + eps),
    )


def conservation_loss(pred, coarse, factor: int):
    """粗い格子ごとの平均が、入力の粗い予報と一致するように促す（整数倍の格子）。"""
    return F.mse_loss(F.avg_pool2d(pred, factor), coarse)


def renormalize(pred, coarse, factor: int):
    """粗い格子ごとの平均が coarse に厳密に一致するよう再スケールする（保存性を厳密にする場合）。"""
    cur = F.avg_pool2d(pred, factor)
    ratio = torch.where(cur > 1e-8, coarse / cur.clamp_min(1e-8), torch.zeros_like(cur))
    return pred * F.interpolate(ratio, scale_factor=factor, mode="nearest")
