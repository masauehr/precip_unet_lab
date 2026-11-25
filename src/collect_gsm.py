"""気象庁GSM日本域(Rjp, 0.1°×0.125°)の1時間降水を、京都大学RISHから取得して沖縄周辺に切り出す（再開可能）。

使い方: /opt/anaconda3/envs/met_env_310/bin/python src/collect_gsm.py [--start 2025-04-01] [--end 2026-11-18]
入力: 00UTC初期値の Lsurf（FD0000-0100）。降水は初期時刻からの積算(mm)なので、隣り合う予報時間の差で1時間降水にする。
保存: data/gsm/YYYYMMDD.npy  形状 (24, 81, 81) float32、mm/h、軸0=予報時間 FT=1..24（FT=k は 初期時刻+k-1〜+k 時間の降水）、
      緯度は北が上（30N→22N）、経度は西から（122E→132E、0.125°間隔なので 81 点）。ファイルは切り出し後に削除。
範囲: 経度 122〜132E・緯度 22〜30N。取得できない日は data/gsm/missing.txt に記録。
データ出典: 気象庁GSM（京都大学RISH配信）。データは再配布しない。
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pygrib
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "gsm"
TMP = OUT / "tmp"
BASE = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; precip-unet-lab/0.1)"}
LAT = (22.0, 30.0)
LON = (122.0, 132.0)


def url_for(d: date) -> tuple:
    name = f"Z__C_RJTD_{d:%Y%m%d}000000_GSM_GPV_Rjp_Gll0p1deg_Lsurf_FD0000-0100_grib2.bin"
    return f"{BASE}/{d:%Y/%m/%d}/{name}", name


def download(d: date) -> Path:
    url, name = url_for(d)
    TMP.mkdir(parents=True, exist_ok=True)
    dest, part = TMP / name, TMP / (name + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    part.rename(dest)
    return dest


def extract(path: Path) -> np.ndarray:
    """積算降水を読み、差分で (24, 81, 81) の1時間降水にして返す。"""
    tp = {}
    lat = lon = None
    grbs = pygrib.open(str(path))
    try:
        for m in grbs:
            if (m.typeOfLevel == "surface" and m.parameterCategory == 1 and m.parameterNumber == 8
                    and int(m.startStep) == 0):
                tp[int(m.endStep)] = np.asarray(m.values, dtype=np.float32)
                if lat is None:
                    la, lo = m.latlons()
                    lat, lon = la[:, 0], lo[0, :]
    finally:
        grbs.close()
    if sorted(tp) != list(range(1, 25)):
        raise ValueError(f"積算降水の予報時間が想定外: {sorted(tp)}")
    acc = np.stack([tp[k] for k in range(1, 25)])
    hourly = np.diff(np.concatenate([np.zeros_like(acc[:1]), acc]), axis=0).clip(min=0)
    # 北が上・経度は西から。範囲を切り出す
    if lat[0] < lat[-1]:
        lat, hourly = lat[::-1], hourly[:, ::-1, :]
    yi = np.where((lat <= LAT[1] + 1e-6) & (lat >= LAT[0] - 1e-6))[0]
    xi = np.where((lon >= LON[0] - 1e-6) & (lon <= LON[1] + 1e-6))[0]
    out = hourly[:, yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1]
    if out.shape != (24, 81, 81):
        raise ValueError(f"切り出し形状が想定外: {out.shape}")
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2026-11-18")
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    n_ok = n_skip = n_ng = 0
    t0 = time.time()
    while d <= end:
        f = OUT / f"{d:%Y%m%d}.npy"
        if f.exists():
            n_skip += 1
        else:
            for k in range(a.retries):
                src = None
                try:
                    src = download(d)
                    arr = extract(src)
                    np.save(f.with_suffix(".tmp.npy"), arr)
                    f.with_suffix(".tmp.npy").rename(f)
                    n_ok += 1
                    print(f"{d} OK  最大 {arr.max():.1f} 総和 {arr.sum():.0f}  経過 {time.time() - t0:.0f}s", flush=True)
                    break
                except requests.HTTPError as e:
                    print(f"{d} 失敗({k + 1}/{a.retries}): HTTP {e.response.status_code}", flush=True)
                    if e.response.status_code == 404:
                        break  # 無い日は再試行しない
                    time.sleep(3)
                except Exception as e:
                    print(f"{d} 失敗({k + 1}/{a.retries}): {str(e)[:80]}", flush=True)
                    time.sleep(3)
                finally:
                    if src is not None:
                        src.unlink(missing_ok=True)  # 自分が取得した一時ファイルだけ消す
            else:
                pass
            if not f.exists():
                n_ng += 1
                with open(OUT / "missing.txt", "a") as m:
                    m.write(f"{d}\n")
        d += timedelta(days=1)
    print(f"完了: 新規 {n_ok} / 取得済み {n_skip} / 失敗 {n_ng}")


if __name__ == "__main__":
    sys.exit(main())
