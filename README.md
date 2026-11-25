# precip_unet_lab

詳しくは [precip-unet-lab.md](precip-unet-lab.md) を参照。

粗い格子の降水量予報を、細かい格子の降水量へ補正する **U-Net**（保存的再格子化 + 対数空間の乗法的補正）を、合成データで学ぶ学習プロジェクト。

## 構成
```
ベース場 = 保存的再格子化(粗い予報)
r        = U-Net(log(ベース場 + 0.1))   # 3×3畳み込み・GroupNorm・SiLU・平均プーリング・双線形拡大・スキップ結合
予測     = ベース場 × exp(r)            # 乗法的補正（0 は 0 のまま）
損失     = 対数空間の MSE（＋任意で保存制約）
評価     = ブートストラップによる 95% 信頼区間
```

## 使い方
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python src/train.py --epochs 20
```

## ファイル
| ファイル | 内容 |
|---|---|
| [src/regrid.py](src/regrid.py) | 保存的再格子化・双線形補間（numpy のみ、割り切れない比も可） |
| [src/synth.py](src/synth.py) | 合成の真の降水場と、誤差を含む粗い予報 |
| [src/model.py](src/model.py) | U-Net・乗法補正・損失・保存制約・再スケール |
| [src/train.py](src/train.py) | 学習と評価（基準比較・ブートストラップ・FSS） |
| [src/boot.py](src/boot.py) | 二段ブートストラップ（学習シード×事例・対の差） |
| [src/prob.py](src/prob.py) | 確率の評価（ブライアススキル・ECE・信頼性図・階級別基準） |
| [src/metrics_ext.py](src/metrics_ext.py) | しきい値別CSI・頻度バイアス・窓別FSS・条件付けの向き別の強雨比・スペクトル |
| src/phase1*.py, phase2*.py, phase3_*.py | 各Phaseの実験スクリプト |
| [PLAN.md](PLAN.md) | 計画と Phase |

状態: Phase 0〜4b 完了（2026-11-25）。
