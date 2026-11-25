"""GSMaP 1時間降水（ガウジ補正 v6）を沖縄周辺で日ごとに取得して保存する（再開可能）。

使い方: ~/projects/jaxa_earth_viewer/.venv/bin/python src/collect_gsmap.py [--start 2025-04-01] [--end 2026-05-10]
保存: data/gsmap/YYYYMMDD.npy  形状 (24, 80, 100) float32、mm/h、時刻は UTC の 0〜23 時（各時の開始）、緯度は北が上。
範囲: 経度 122〜132E・緯度 22〜30N、0.1°。取得済みの日はスキップ。取得できない日は data/gsmap/missing.txt に記録。
データは再配布しない。利用時は JAXA のクレジット表記が必要（GSMaP: JAXA/EORC）。
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from jaxa.earth import je

OUT = Path(__file__).resolve().parents[1] / "data" / "gsmap"
COLLECTION = "JAXA.EORC_GSMaP_standard.Gauge.v6_hourly"
BBOX = [122, 22, 132, 30]


def fetch_day(d: date) -> np.ndarray:
    """1日分（24時間）を (24, 80, 100) で返す。"""
    data = je.ImageCollection(collection=COLLECTION, ssl_verify=True) \
        .filter_date(dlim=[f"{d}T00:00:00", f"{d}T23:00:00"]) \
        .filter_resolution(ppu=10).filter_bounds(bbox=BBOX).select(band="PRECIP").get_images()
    arr = np.array(je.ImageProcess(data).raster.img, dtype=np.float32)[..., 0]
    if arr.shape != (24, 80, 100):
        raise ValueError(f"形状が想定外: {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2026-05-10")
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
                try:
                    arr = fetch_day(d)
                    np.save(f.with_suffix(".tmp.npy"), arr)
                    f.with_suffix(".tmp.npy").rename(f)
                    n_ok += 1
                    print(f"{d} OK  最大 {np.nanmax(arr):.1f} NaN {int(np.isnan(arr).sum())}  経過 {time.time() - t0:.0f}s", flush=True)
                    break
                except Exception as e:
                    print(f"{d} 失敗({k + 1}/{a.retries}): {str(e)[:80]}", flush=True)
                    time.sleep(3)
            else:
                n_ng += 1
                with open(OUT / "missing.txt", "a") as m:
                    m.write(f"{d}\n")
        d += timedelta(days=1)
    print(f"完了: 新規 {n_ok} / 取得済み {n_skip} / 失敗 {n_ng}")


if __name__ == "__main__":
    sys.exit(main())
