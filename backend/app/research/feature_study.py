"""특징별 Overnight 예측력 검증 — "어떤 특징을 가진 종목을 장 마감 전에 샀을 때 다음날 성과가 좋았는가?"

미리 효과를 가정하지 않는다. 각 특징에 대해:
- 종목 특징: 일별 횡단면 Spearman IC (특징 순위 vs 다음날 종가/시가 수익률 순위), 5분위 스프레드,
  최상위 분위 매수 시 비용 차감 평균 수익
- 시장 특징(m_): 일별 시계열 상관 (특징 값 vs 그날 전체 종목 평균 수익)
- 학습 구간(in-sample)과 이후 구간(Out-of-Sample)을 분리해서 각각 계산하고,
  OOS 에서도 같은 방향·충분한 t-stat 일 때만 '유효'로 판정.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..prediction.model import FEATURE_GROUP, FEATURE_LABEL, FEATURE_NAMES


def _t(x: pd.Series):
    x = x.dropna()
    if len(x) < 5 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def _daily_ic(df: pd.DataFrame, feat: str, target: str) -> pd.Series:
    d = df[["trade_date", feat, target]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    d = d.assign(rf=d.groupby("trade_date")[feat].rank(), rt=d.groupby("trade_date")[target].rank())
    g = d.groupby("trade_date")
    return g.apply(lambda x: x.rf.corr(x.rt) if len(x) >= 10 and x.rf.nunique() > 2 else np.nan,
                   include_groups=False).dropna()


def _quintiles(df: pd.DataFrame, feat: str) -> dict:
    d = df[["trade_date", feat, "close_ret", "open_ret"]].dropna(subset=[feat, "close_ret"])
    if len(d) < 50:
        return {}
    q = d.groupby("trade_date")[feat].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
                                                if len(s) >= 10 else np.nan)
    d = d.assign(q=q).dropna(subset=["q"])
    by = d.groupby("q").close_ret.mean()
    if len(by) < 5:
        return {}
    cost = settings.round_trip_cost
    return {"q1_close": round(float(by.iloc[0]), 5), "q5_close": round(float(by.iloc[-1]), 5),
            "spread_close": round(float(by.iloc[-1] - by.iloc[0]), 5),
            "q5_net_close": round(float(by.iloc[-1] - cost), 5),
            "q5_open": round(float(d[d.q == 4].open_ret.mean()), 5)}


def _ts_corr(df: pd.DataFrame, feat: str) -> tuple[float | None, int]:
    daily = df.groupby("trade_date").agg(x=(feat, "first"), y=("close_ret", "mean")).dropna()
    if len(daily) < 15 or daily.x.nunique() < 3:
        return None, len(daily)
    return float(daily.x.corr(daily.y)), len(daily)


def study(samples: pd.DataFrame, oos_frac: float = 0.4, oos_start=None) -> dict:
    if samples.empty:
        return {"message": "표본 없음"}
    dates = sorted(samples.trade_date.unique())
    if oos_start is None:
        oos_start = dates[int(len(dates) * (1 - oos_frac))] if len(dates) > 10 else dates[-1]
    ins, oos = samples[samples.trade_date < oos_start], samples[samples.trade_date >= oos_start]
    out = []
    for f in FEATURE_NAMES:
        if f not in samples or samples[f].notna().mean() < 0.05:
            out.append({"feature": f, "label": FEATURE_LABEL[f], "group": FEATURE_GROUP[f], "verdict": "데이터 없음",
                        "coverage": 0.0})
            continue
        rec = {"feature": f, "label": FEATURE_LABEL[f], "group": FEATURE_GROUP[f],
               "coverage": round(float(samples[f].notna().mean()), 3)}
        if f.startswith("m_"):
            ci, ni = _ts_corr(ins, f)
            co, no = _ts_corr(oos, f)
            rec.update(kind="시장(시계열 상관)", ic_in=_r(ci), ic_oos=_r(co), n_days_in=ni, n_days_oos=no,
                       t_in=_r(ci * np.sqrt(max(ni - 2, 1)) / np.sqrt(max(1e-9, 1 - ci ** 2))) if ci is not None else None,
                       t_oos=_r(co * np.sqrt(max(no - 2, 1)) / np.sqrt(max(1e-9, 1 - co ** 2))) if co is not None else None)
        else:
            ii, io = _daily_ic(ins, f, "close_ret"), _daily_ic(oos, f, "close_ret")
            gi = _daily_ic(oos, f, "open_ret")
            rec.update(kind="종목(횡단면 IC)", ic_in=_r(ii.mean()), t_in=_r(_t(ii)), ic_oos=_r(io.mean()), t_oos=_r(_t(io)),
                       ic_oos_open=_r(gi.mean()), t_oos_open=_r(_t(gi)), n_days_in=int(len(ii)), n_days_oos=int(len(io)),
                       quintile_oos=_quintiles(oos, f))
            if "trend" in oos:
                rec["ic_oos_by_trend"] = {k: _r(_daily_ic(g, f, "close_ret").mean()) for k, g in oos.groupby("trend")
                                          if g.trade_date.nunique() >= 10}
        rec["verdict"] = verdict(rec)
        out.append(rec)
    out.sort(key=lambda r: -(abs(r.get("t_oos") or 0)))
    return {"in_sample": [str(dates[0]), str(ins.trade_date.max()) if not ins.empty else None],
            "out_of_sample": [str(oos_start), str(dates[-1])], "n_samples": int(len(samples)),
            "n_days": len(dates), "features": out,
            "method": "IC = 일별 순위상관(특징 vs 다음날 종가수익). 유효 = 학습구간 |t|≥2 이고 OOS 같은 부호 |t|≥1.5"}


def verdict(r: dict) -> str:
    ti, to = r.get("t_in"), r.get("t_oos")
    if ti is None or to is None:
        return "표본 부족"
    if abs(ti) >= 2 and abs(to) >= 1.5 and np.sign(ti) == np.sign(to):
        return "OOS 유효 (+)" if to > 0 else "OOS 유효 (−, 역방향)"
    if abs(ti) >= 2 and np.sign(ti) != np.sign(to):
        return "불안정 (OOS 에서 방향 반전)"
    if abs(ti) >= 2:
        return "학습구간만 유효 (OOS 미확인)"
    if abs(to) >= 2:
        return "OOS 에서만 관찰 (추가 검증 필요)"
    return "효과 없음"


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
