"""収集した GSMaP / GSM の件数と降水強度の分布を月別に集計する。

使い方: .venv/bin/python src/data_summary.py
有効な日 = 両方そろい、GSMaP が全面 -99 でない日。画素単位で GSMaP の欠測(NaN)は除く。
"""
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "data"
THRS = (1, 5, 10, 20)


def valid_days():
    days = []
    for p in sorted((D / "gsmap").glob("2*.npy")):
        if not (D / "gsm" / p.name).exists():
            continue
        if (np.load(p) < -90).all():
            continue  # 全面 -99 の日
        days.append(p.stem)
    return days


if __name__ == "__main__":
    days = valid_days()
    months = sorted({d[:6] for d in days})
    print(f"有効な日 {len(days)}（{days[0]}〜{days[-1]}）")
    hdr = f"{'月':<7}{'日数':>4}{'平均mm/h':>9}" + "".join(f"{'≥' + str(t) + ' 画素割合%':>13}" for t in THRS) + \
          "".join(f"{'≥' + str(t) + ' 日数':>9}" for t in (10, 20)) + f"{'GSM平均':>9}"
    print(hdr)
    tot = {t: 0 for t in THRS}
    n_px = 0
    rows = []
    for m in months:
        ds = [d for d in days if d.startswith(m)]
        gm = np.stack([np.where(np.load(D / "gsmap" / f"{d}.npy") < 0, np.nan, np.load(D / "gsmap" / f"{d}.npy")) for d in ds])
        gs = np.stack([np.load(D / "gsm" / f"{d}.npy") for d in ds])
        ok = np.isfinite(gm)
        v = gm[ok]
        fr = [100 * (v >= t).mean() for t in THRS]
        dmax = np.nanmax(gm.reshape(len(ds), -1), axis=1)
        for t in THRS:
            tot[t] += int((v >= t).sum())
        n_px += v.size
        print(f"{m:<7}{len(ds):>4}{v.mean():>9.3f}" + "".join(f"{x:>13.2f}" for x in fr)
              + f"{int((dmax >= 10).sum()):>9}{int((dmax >= 20).sum()):>9}{gs.mean():>9.3f}")
    print(f"\n全期間 画素数 {n_px:,}")
    for t in THRS:
        print(f"  ≥{t:>2} mm/h の画素 {tot[t]:>10,}（{100 * tot[t] / n_px:.2f}%）")
