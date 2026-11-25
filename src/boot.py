"""二段ブートストラップ（学習シード × テスト事例）。numpy のみ。

学習のたびに結果がぶれるので、「テスト事例のばらつき」だけの信頼区間は狭すぎる。
  事例のみ : シードは固定し、各シードのテスト事例を再標本化（これまでの表の区間に相当）
  シードのみ: 事例は固定し、学習シードを再標本化
  二段     : シードを再標本化し、選んだシードの中で事例を再標本化（学習のぶれを含む）
2つの設定 A・B を比べるときは、同じシード・同じ事例で対にして差を取る（対の再標本化）。

事例ごとに「部分和」を持つ（parts）。指標は、再標本化した事例の部分和から求める（比の指標も正しく出る）。
  rmse   : 事例ごとの対数RMSE（平均を取る）
  sp, st : 予測・真値の総量     → total  = log(Σsp / Σst)
  sps, sts: 強雨画素の予測・真値の総量 → strong = log(Σsps / Σsts)
  wsq, wn: 雨のある画素の対数誤差の2乗和・画素数 → wet_rmse = sqrt(Σwsq / Σwn)
  fnum, fden: FSS の分子・分母（5mm/h） → fss5 = 1 − Σfnum / Σfden
"""
import numpy as np

KEYS = ["rmse", "sp", "st", "sps", "sts", "wsq", "wn", "fnum", "fden"]
STATS = ["rmse", "total", "strong", "wet_rmse", "fss5"]


def stats_from_sums(p: dict) -> dict:
    """部分和（事例ごとの配列 or その合計）から指標を求める。rmse は配列なら平均を取る。"""
    g = lambda k: np.sum(p[k])
    return dict(
        rmse=float(np.mean(p["rmse"])),
        total=float(np.log(max(g("sp"), 1e-9) / g("st"))),
        strong=float(np.log(max(g("sps"), 1e-9) / g("sts"))),
        wet_rmse=float(np.sqrt(g("wsq") / g("wn"))),
        fss5=float(1 - g("fnum") / g("fden")),
    )


def case_index(rng, n: int, block: int = 1) -> np.ndarray:
    """n 件の事例の再標本化インデックス。block>1 なら連続 block 件のまとまりで（時間相関がある実データ向け）。"""
    nb = max(n // block, 1)
    starts = rng.integers(0, nb, size=nb) * block
    return (starts[:, None] + np.arange(block)).ravel()


def _gather(parts_seeds, seed_ids, case_idx_per_seed):
    """選んだシード・事例の部分を連結する。"""
    return {
        k: np.concatenate([parts_seeds[s][k][ci] for s, ci in zip(seed_ids, case_idx_per_seed)])
        for k in KEYS
    }


def diff_ci(A, B, mode: str = "two_level", n_boot: int = 2000, seed: int = 0, block: int = 1):
    """A−B の各指標の (点推定, 下限, 上限)。A・B は「シードごとの parts の配列」で、同じシード・同じ事例で対。

    mode: "cases"（事例のみ）/ "seeds"（シードのみ）/ "two_level"（二段）
    """
    rng = np.random.default_rng(seed)
    S = len(A)
    n = len(A[0]["rmse"])
    point = {k: stats_from_sums(_gather(A, range(S), [np.arange(n)] * S))[k]
             - stats_from_sums(_gather(B, range(S), [np.arange(n)] * S))[k] for k in STATS}
    draws = {k: np.empty(n_boot) for k in STATS}
    for b in range(n_boot):
        seed_ids = range(S) if mode == "cases" else rng.integers(0, S, size=S)
        cis = [np.arange(n) if mode == "seeds" else case_index(rng, n, block) for _ in range(S)]
        sa = stats_from_sums(_gather(A, seed_ids, cis))
        sb = stats_from_sums(_gather(B, seed_ids, cis))
        for k in STATS:
            draws[k][b] = sa[k] - sb[k]
    return {k: (point[k], *np.percentile(draws[k], [2.5, 97.5])) for k in STATS}
